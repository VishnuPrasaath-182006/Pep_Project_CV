"""
Explainable AI (XAI) & Quantitative Validation Module
Implements LayerCAM attribution (Jiang et al., 2021) for the CASR residual adapter,
and rigorous ground-truth evaluation when physical masks are uploaded.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compute_layercam(adapter, features_tensor, candidate_mask_tensor):
    """
    Computes audited LayerCAM attribution on CASR adapter layer block2.
    Target objective: Flood-class residual score within candidate regions:
        S = sum( DeltaZ[:, 1:2] * C )
    Gradients are calculated strictly for this pass; zero parameter drift is guaranteed.
    """
    activations = {}
    
    def forward_hook(module, inp, out):
        activations["feat"] = out
        
    hook = adapter.block2.register_forward_hook(forward_hook)
    
    # Enable gradient calculation exclusively inside attribution block
    with torch.enable_grad():
        # Ensure input tensors do not require grad themselves
        feat_in = features_tensor.detach().clone().requires_grad_(False)
        cand_in = candidate_mask_tensor.detach().clone().requires_grad_(False)
        
        # Temporarily enable requires_grad on adapter parameters
        orig_grad_states = {}
        for n, p in adapter.named_parameters():
            orig_grad_states[n] = p.requires_grad
            p.requires_grad = True
            
        adapter.zero_grad()
        
        delta_z, gate, res_update = adapter(feat_in, cand_in)
        act = activations["feat"]
        act.retain_grad()
        
        # Target: Flood residual score (channel 1) within candidate mask
        flood_score = (delta_z[:, 1:2] * cand_in).sum()
        flood_score.backward()
        
        grad = act.grad.detach()
        act_val = act.detach()
        hook.remove()
        
        # Restore original parameter grad states and zero gradients
        for n, p in adapter.named_parameters():
            p.requires_grad = orig_grad_states.get(n, False)
        adapter.zero_grad()
        
    # LayerCAM formulation: relu(grad) * act
    weights = torch.clamp(grad, min=0.0)
    cam = torch.sum(weights * act_val, dim=1, keepdim=True)
    cam = torch.clamp(cam, min=0.0)
    
    # Bilinear interpolation back to input patch dimensions
    _, _, h, w = features_tensor.shape
    cam_interp = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
    
    # Min-max normalization into [0.0, 1.0]
    c_min = cam_interp.min()
    c_max = cam_interp.max()
    if c_max > c_min:
        cam_norm = (cam_interp - c_min) / (c_max - c_min + 1e-8)
    else:
        cam_norm = torch.zeros_like(cam_interp)
        
    return cam_norm.squeeze().cpu().numpy()


def compute_groundtruth_evaluation(pred_mask, gt_mask, valid_mask):
    """
    Computes rigorous confusion matrix and classification metrics against ground-truth labels.
    
    Ground-truth mapping:
        0: Background / Land
        1: Flood Inundated Water
       -1: Ignore / Invalid / Nodata
    """
    eval_mask = valid_mask & (gt_mask != -1)
    
    tp = int(((pred_mask == 1) & (gt_mask == 1) & eval_mask).sum())
    fp = int(((pred_mask == 1) & (gt_mask == 0) & eval_mask).sum())
    fn = int(((pred_mask == 0) & (gt_mask == 1) & eval_mask).sum())
    tn = int(((pred_mask == 0) & (gt_mask == 0) & eval_mask).sum())
    
    total_valid = tp + fp + fn + tn
    denom_iou = tp + fp + fn
    flood_iou = (tp / denom_iou * 100.0) if denom_iou > 0 else (100.0 if denom_iou == 0 else 0.0)
    
    denom_dice = 2 * tp + fp + fn
    dice = (2.0 * tp / denom_dice * 100.0) if denom_dice > 0 else (100.0 if denom_dice == 0 else 0.0)
    
    denom_prec = tp + fp
    precision = (tp / denom_prec * 100.0) if denom_prec > 0 else 0.0
    
    denom_rec = tp + fn
    recall = (tp / denom_rec * 100.0) if denom_rec > 0 else 0.0
    
    denom_spec = tn + fp
    specificity = (tn / denom_spec * 100.0) if denom_spec > 0 else 0.0
    
    accuracy = ((tp + tn) / total_valid * 100.0) if total_valid > 0 else 0.0
    
    # Construct 4-Color Confusion Error Map
    h, w = pred_mask.shape
    error_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # TN: Dark Slate Blue [30, 41, 59]
    error_rgb[(pred_mask == 0) & (gt_mask == 0) & eval_mask] = [30, 41, 59]
    # TP: Vibrant Emerald Green [16, 185, 129]
    error_rgb[(pred_mask == 1) & (gt_mask == 1) & eval_mask] = [16, 185, 129]
    # FP: Coral Crimson [239, 68, 68] (False alarms)
    error_rgb[(pred_mask == 1) & (gt_mask == 0) & eval_mask] = [239, 68, 68]
    # FN: Amber Yellow [245, 158, 11] (Missed floods)
    error_rgb[(pred_mask == 0) & (gt_mask == 1) & eval_mask] = [245, 158, 11]
    
    error_img = Image.fromarray(error_rgb)
    
    metrics = {
        "flood_iou": round(flood_iou, 2),
        "flood_dice": round(dice, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "specificity": round(specificity, 2),
        "pixel_accuracy": round(accuracy, 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evaluated_pixels": total_valid
    }
    
    return metrics, error_img
