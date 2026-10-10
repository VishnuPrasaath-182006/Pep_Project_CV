"""
Stage 6: Validation XAI & Qualitative Diagnostics
Generates spatial diagnostic maps, LayerCAM class activation maps, error comparisons,
and gate analysis on representative validation scenes from Bolivia and US-Dakota.
"""

import sys
import os
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from config import load_config
from model import build_model
from casr import CASRAdapter, CAVEFloodModel
from evaluate_source_chips import compute_binary_metrics


def compute_layercam(model_adapter, features_tensor, candidate_mask_tensor):
    """
    Computes LayerCAM on the CASR block2 representation for the flood class.
    w_k(i, j) = relu(dScore / d A_k(i, j))
    CAM(i, j) = relu( sum_k w_k(i, j) * A_k(i, j) )
    """
    activations = {}
    def hook_fn(module, inp, out):
        activations['feat'] = out
        
    hook = model_adapter.block2.register_forward_hook(hook_fn)
    
    orig_requires_grad = {p: p.requires_grad for p in model_adapter.parameters()}
    try:
        # Enable gradients for adapter parameters temporarily for backward graph construction
        for p in model_adapter.parameters():
            p.requires_grad = True
        model_adapter.zero_grad()
            
        # Forward pass
        delta_z, gate, res_update = model_adapter(features_tensor, candidate_mask_tensor)
        if 'feat' in activations:
            activations['feat'].retain_grad()
            
        # Flood score (target class = 1)
        # Target gradient on the flood logit delta within candidate regions
        flood_score = (delta_z[:, 1:2] * candidate_mask_tensor).sum()
        flood_score.backward(retain_graph=False)
        
        act = activations['feat'].detach()
        grad = activations['feat'].grad.detach()
    finally:
        hook.remove()
        model_adapter.zero_grad()
        for p, req in orig_requires_grad.items():
            p.requires_grad = req
    
    # LayerCAM formula
    weights = torch.relu(grad)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = F.interpolate(cam, size=(512, 512), mode='bilinear', align_corners=False)
    
    # Normalize to [0, 1]
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    else:
        cam = torch.zeros_like(cam)
        
    return cam.squeeze().cpu().numpy()


def make_rgb_composite(raw_img_4b):
    """
    Creates normalized RGB display image from 4-band PlanetScope imagery (B0=Red, B1=Green, B2=Blue).
    Applies 2-98 percentile contrast stretch.
    """
    rgb = raw_img_4b[:3].transpose(1, 2, 0).astype(np.float32)
    # Check valid pixels
    valid = (raw_img_4b != 0).any(axis=0)
    if valid.sum() == 0:
        return np.zeros_like(rgb)
        
    p2, p98 = np.percentile(rgb[valid], (2, 98), axis=0)
    rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    rgb_stretched[~valid] = 0.0
    return rgb_stretched


def make_error_map(y_true, y_pred, valid_mask):
    """
    Creates RGB error map:
    TP = Green ([0.1, 0.8, 0.2])
    TN = Dark Gray ([0.2, 0.2, 0.25])
    FP = Red ([0.9, 0.2, 0.2])
    FN = Amber/Orange ([1.0, 0.65, 0.0])
    Invalid = Black ([0.0, 0.0, 0.0])
    """
    h, w = y_true.shape
    err_rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # Default dark gray for valid TN
    tp = (y_true == 1) & (y_pred == 1) & valid_mask
    tn = (y_true == 0) & (y_pred == 0) & valid_mask
    fp = (y_true == 0) & (y_pred == 1) & valid_mask
    fn = (y_true == 1) & (y_pred == 0) & valid_mask
    
    err_rgb[tn] = [0.2, 0.2, 0.25]
    err_rgb[tp] = [0.1, 0.8, 0.2]
    err_rgb[fp] = [0.9, 0.2, 0.2]
    err_rgb[fn] = [1.0, 0.65, 0.0]
    
    return err_rgb


def run_stage6():
    print("=" * 70)
    print("STAGE 6: VALIDATION XAI & QUALITATIVE DIAGNOSTICS")
    print("=" * 70)
    
    config = load_config()
    val_dir = Path("artifacts/CAVE_Flood_Antigravity/cache/candidate_regions/val")
    val_files = sorted(list(val_dir.glob("*.npz")))
    
    out_fig_dir = Path("artifacts/CAVE_Flood_Antigravity/figures")
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load E2 checkpoint
    ckpt_e2_path = "UNet-Mamba/checkpoints/best_e2_model.pth"
    ckpt_e2 = torch.load(ckpt_e2_path, weights_only=True)
    
    adapter_e2 = CASRAdapter(in_channels=9, hidden_channels=32, num_classes=2, lambda_res=1.0)
    adapter_e2.load_state_dict(ckpt_e2["casr_state_dict"])
    adapter_e2.eval()
    
    # Group patches by scene
    scenes = {}
    for f in val_files:
        stem_key = f.stem.rsplit("_p", 1)[0]
        p_idx = int(f.stem.rsplit("_p", 1)[1])
        if stem_key not in scenes:
            scenes[stem_key] = {}
        scenes[stem_key][p_idx] = f
        
    quadrant_coords = {0: (0, 0), 1: (0, 512), 2: (512, 0), 3: (512, 512)}
    norm_mean = np.array(config["data"]["norm_mean"], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config["data"]["norm_std"], dtype=np.float32).reshape(4, 1, 1)
    
    # Representative scenes to diagnose
    target_scenes = [
        "Bolivia_BOL_1176",      # Highest Bolivia IoU gain (+4.51%)
        "US-Dakota_RRN_10_46",   # Highest US-Dakota IoU gain (+1.30%)
        "Bolivia_BOL_1123"       # Challenging boundary scene (+4.20%)
    ]
    
    summary_report = {
        "experiment": "E2",
        "checkpoint": ckpt_e2_path,
        "diagnosed_scenes": []
    }
    
    for scene_id in target_scenes:
        if scene_id not in scenes or len(scenes[scene_id]) != 4:
            print(f"Warning: Scene {scene_id} incomplete or not found.")
            continue
            
        print(f"\nProcessing Diagnostic Visualization for: {scene_id}...")
        p_map = scenes[scene_id]
        
        # 1024x1024 canvases
        full_img = np.zeros((4, 1024, 1024), dtype=np.int64)
        full_mask = np.zeros((1024, 1024), dtype=np.int64)
        full_cand_mask = np.zeros((1024, 1024), dtype=np.float32)
        full_entropy = np.zeros((1024, 1024), dtype=np.float32)
        full_grad = np.zeros((1024, 1024), dtype=np.float32)
        full_guidance = np.zeros((1024, 1024), dtype=np.float32)
        full_e0_logits = np.zeros((2, 1024, 1024), dtype=np.float32)
        full_e2_logits = np.zeros((2, 1024, 1024), dtype=np.float32)
        full_gate = np.zeros((1024, 1024), dtype=np.float32)
        full_res_mag = np.zeros((1024, 1024), dtype=np.float32)
        full_layercam = np.zeros((1024, 1024), dtype=np.float32)
        
        for p_idx in range(4):
            r, c = quadrant_coords[p_idx]
            p_path = p_map[p_idx]
            data = np.load(p_path)
            
            raw_img = data["raw_image"]
            raw_mask = data["raw_mask"]
            cand_mask = data["candidate_mask"]
            flood_prob = data["flood_prob"]
            entropy_map = data["entropy_map"]
            grad_map = data["gradient_map"]
            guidance = data["guidance_e2"]
            base_logits = data["patch_logits"]
            
            full_img[:, r:r+512, c:c+512] = raw_img
            full_mask[r:r+512, c:c+512] = raw_mask
            full_cand_mask[r:r+512, c:c+512] = cand_mask
            full_entropy[r:r+512, c:c+512] = entropy_map
            full_grad[r:r+512, c:c+512] = grad_map
            full_guidance[r:r+512, c:c+512] = guidance
            full_e0_logits[:, r:r+512, c:c+512] = base_logits
            
            # Feature assembly
            img_norm = (raw_img.astype(np.float32) - norm_mean) / (norm_std + 1e-6)
            feats = np.concatenate([
                img_norm,
                flood_prob[np.newaxis, ...],
                entropy_map[np.newaxis, ...],
                grad_map[np.newaxis, ...],
                cand_mask[np.newaxis, ...],
                guidance[np.newaxis, ...]
            ], axis=0).astype(np.float32)
            
            feat_t = torch.from_numpy(feats).unsqueeze(0).float()
            cand_t = torch.from_numpy(cand_mask).unsqueeze(0).unsqueeze(0).float()
            
            # LayerCAM calculation on this patch
            patch_cam = compute_layercam(adapter_e2, feat_t.clone(), cand_t.clone())
            full_layercam[r:r+512, c:c+512] = patch_cam
            
            # Forward pass
            with torch.no_grad():
                delta_z, gate, res_update = adapter_e2(feat_t, cand_t)
                z_final = torch.from_numpy(base_logits).unsqueeze(0) + res_update
                
                full_e2_logits[:, r:r+512, c:c+512] = z_final.squeeze(0).numpy()
                full_gate[r:r+512, c:c+512] = gate.squeeze().numpy()
                
                # Residual magnitude: ||Delta Z||_2
                res_mag = torch.norm(res_update.squeeze(0), dim=0).numpy()
                full_res_mag[r:r+512, c:c+512] = res_mag
                
        # Valid pixels
        valid_mask = (full_img != 0).any(axis=0) & (full_mask != -1)
        y_true = full_mask
        
        # Probabilities & Binary predictions
        e0_probs = np.exp(full_e0_logits[1] - np.logaddexp(full_e0_logits[0], full_e0_logits[1]))
        e2_probs = np.exp(full_e2_logits[1] - np.logaddexp(full_e2_logits[0], full_e2_logits[1]))
        
        e0_pred = np.argmax(full_e0_logits, axis=0)
        e2_pred = np.argmax(full_e2_logits, axis=0)
        
        # Metrics
        def calc_scene_metrics(pred):
            v_true = y_true[valid_mask]
            v_pred = pred[valid_mask]
            tp = int(((v_true == 1) & (v_pred == 1)).sum())
            fp = int(((v_true == 0) & (v_pred == 1)).sum())
            fn = int(((v_true == 1) & (v_pred == 0)).sum())
            tn = int(((v_true == 0) & (v_pred == 0)).sum())
            return compute_binary_metrics(tp, fp, fn, tn), tp, fp, fn, tn
            
        m0, tp0, fp0, fn0, tn0 = calc_scene_metrics(e0_pred)
        m2, tp2, fp2, fn2, tn2 = calc_scene_metrics(e2_pred)
        
        fp_reduction = fp0 - fp2
        fn_reduction = fn0 - fn2
        
        print(f"  E0 Flood IoU: {m0['flood_iou']:.4f} -> E2: {m2['flood_iou']:.4f} (Delta: {m2['flood_iou'] - m0['flood_iou']:+.4f})")
        print(f"  FP: {fp0} -> {fp2} (Change: {-fp_reduction:+d}) | FN: {fn0} -> {fn2} (Change: {-fn_reduction:+d})")
        
        # Mean gate inside candidate regions vs outside
        mean_gate_cand = float(full_gate[full_cand_mask == 1.0].mean()) if (full_cand_mask == 1.0).sum() > 0 else 0.0
        max_res_mag = float(full_res_mag.max())
        
        scene_info = {
            "scene_id": scene_id,
            "e0_flood_iou": m0['flood_iou'],
            "e2_flood_iou": m2['flood_iou'],
            "delta_iou": m2['flood_iou'] - m0['flood_iou'],
            "e0_precision": m0['flood_precision'],
            "e2_precision": m2['flood_precision'],
            "e0_recall": m0['flood_recall'],
            "e2_recall": m2['flood_recall'],
            "fp_initial": fp0,
            "fp_final": fp2,
            "fp_reduction": fp_reduction,
            "fn_initial": fn0,
            "fn_final": fn2,
            "fn_reduction": fn_reduction,
            "mean_gate_in_candidate": mean_gate_cand,
            "max_residual_magnitude": max_res_mag
        }
        summary_report["diagnosed_scenes"].append(scene_info)
        
        # -------------------------------------------------------------
        # Generate 12-Panel Comprehensive Diagnostic Figure
        # -------------------------------------------------------------
        fig, axes = plt.subplots(3, 4, figsize=(22, 16.5))
        plt.subplots_adjust(wspace=0.15, hspace=0.22)
        fig.suptitle(
            f"CAVE-Flood XAI & Spatial Diagnostics: {scene_id}\n"
            f"Baseline E0 IoU: {m0['flood_iou']:.4f}  -->  E2 CASR IoU: {m2['flood_iou']:.4f}  "
            f"(Delta: {m2['flood_iou'] - m0['flood_iou']:+.4f} | FP delta: {-fp_reduction:+d}, FN delta: {-fn_reduction:+d})",
            fontsize=16, fontweight='bold', y=0.98
        )
        
        rgb_disp = make_rgb_composite(full_img)
        err_e0 = make_error_map(y_true, e0_pred, valid_mask)
        err_e2 = make_error_map(y_true, e2_pred, valid_mask)
        
        # Difference / Correction Map
        edit_map = np.zeros((1024, 1024, 3), dtype=np.float32)
        flipped_to_flood = (e0_pred == 0) & (e2_pred == 1) & valid_mask
        flipped_to_bg = (e0_pred == 1) & (e2_pred == 0) & valid_mask
        edit_map[flipped_to_flood] = [0.0, 0.8, 1.0]   # Cyan: newly added flood
        edit_map[flipped_to_bg] = [1.0, 0.2, 0.8]      # Magenta: suppressed false flood
        edit_map[(e0_pred == e2_pred) & valid_mask] = [0.15, 0.15, 0.2]  # Dark blue-gray: untouched
        
        # Panels:
        # Row 0:
        # (0, 0): RGB Imagery
        axes[0, 0].imshow(rgb_disp)
        axes[0, 0].set_title("(a) PlanetScope RGB (Bands 1-3)", fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        
        # (0, 1): Ground Truth Mask
        gt_disp = np.ma.masked_where(~valid_mask, y_true)
        im_gt = axes[0, 1].imshow(gt_disp, cmap='Blues', vmin=0, vmax=1)
        axes[0, 1].set_facecolor('black')
        axes[0, 1].set_title("(b) Ground Truth (1=Flood, 0=Non-Flood)", fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im_gt, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # (0, 2): Baseline E0 Prediction
        pred0_disp = np.ma.masked_where(~valid_mask, e0_pred)
        im_p0 = axes[0, 2].imshow(pred0_disp, cmap='Blues', vmin=0, vmax=1)
        axes[0, 2].set_facecolor('black')
        axes[0, 2].set_title(f"(c) Baseline E0 Pred (IoU: {m0['flood_iou']:.4f})", fontsize=11, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im_p0, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # (0, 3): E2 Refined Prediction
        pred2_disp = np.ma.masked_where(~valid_mask, e2_pred)
        im_p2 = axes[0, 3].imshow(pred2_disp, cmap='Blues', vmin=0, vmax=1)
        axes[0, 3].set_facecolor('black')
        axes[0, 3].set_title(f"(d) CASR E2 Refined Pred (IoU: {m2['flood_iou']:.4f})", fontsize=11, fontweight='bold')
        axes[0, 3].axis('off')
        plt.colorbar(im_p2, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Row 1:
        # (1, 0): Predictive Entropy H
        ent_disp = np.ma.masked_where(~valid_mask, full_entropy)
        im_ent = axes[1, 0].imshow(ent_disp, cmap='inferno')
        axes[1, 0].set_facecolor('black')
        axes[1, 0].set_title("(e) Predictive Entropy H(x)", fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im_ent, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # (1, 1): Spatial Boundary Gradient ||\nabla P||
        grad_disp = np.ma.masked_where(~valid_mask, full_grad)
        im_gr = axes[1, 1].imshow(grad_disp, cmap='magma')
        axes[1, 1].set_facecolor('black')
        axes[1, 1].set_title(r"(f) Boundary Gradient $||\nabla P||$", fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im_gr, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # (1, 2): Selected Candidate Mask C
        cand_disp = np.ma.masked_where(~valid_mask, full_cand_mask)
        im_c = axes[1, 2].imshow(cand_disp, cmap='copper', vmin=0, vmax=1)
        axes[1, 2].set_facecolor('black')
        axes[1, 2].set_title("(g) Selected Candidate Tiles C (Top-4/16)", fontsize=11, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im_c, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # (1, 3): RemoteCLIP Vision-Language Guidance g
        guid_disp = np.ma.masked_where(~valid_mask | (full_cand_mask == 0), full_guidance)
        im_g = axes[1, 3].imshow(guid_disp, cmap='coolwarm', vmin=-1.5, vmax=1.5)
        axes[1, 3].set_facecolor('black')
        axes[1, 3].set_title("(h) RemoteCLIP Guidance g", fontsize=11, fontweight='bold')
        axes[1, 3].axis('off')
        plt.colorbar(im_g, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        # Row 2:
        # (2, 0): CASR Gate G
        gate_disp = np.ma.masked_where(~valid_mask | (full_cand_mask == 0), full_gate)
        im_gate = axes[2, 0].imshow(gate_disp, cmap='viridis', vmin=0, vmax=1)
        axes[2, 0].set_facecolor('black')
        axes[2, 0].set_title(f"(i) CASR Gate G (Cand Mean: {mean_gate_cand:.2f})", fontsize=11, fontweight='bold')
        axes[2, 0].axis('off')
        plt.colorbar(im_gate, ax=axes[2, 0], fraction=0.046, pad=0.04)
        
        # (2, 1): Residual Correction Magnitude ||Delta Z||
        res_disp = np.ma.masked_where(~valid_mask | (full_cand_mask == 0), full_res_mag)
        im_res = axes[2, 1].imshow(res_disp, cmap='plasma', vmin=0, vmax=max(0.5, max_res_mag))
        axes[2, 1].set_facecolor('black')
        axes[2, 1].set_title(r"(j) Residual Update $||\Delta Z_{\mathrm{res}}||_2$", fontsize=11, fontweight='bold')
        axes[2, 1].axis('off')
        plt.colorbar(im_res, ax=axes[2, 1], fraction=0.046, pad=0.04)
        
        # (2, 2): LayerCAM Saliency for Flood Class
        cam_disp = np.ma.masked_where(~valid_mask | (full_cand_mask == 0), full_layercam)
        im_cam = axes[2, 2].imshow(cam_disp, cmap='turbo', vmin=0, vmax=1)
        axes[2, 2].set_facecolor('black')
        axes[2, 2].set_title("(k) LayerCAM Flood Saliency", fontsize=11, fontweight='bold')
        axes[2, 2].axis('off')
        plt.colorbar(im_cam, ax=axes[2, 2], fraction=0.046, pad=0.04)
        
        # (2, 3): Edit / Modification Map (Cyan=Added, Magenta=Removed)
        axes[2, 3].imshow(edit_map)
        axes[2, 3].set_title(f"(l) CASR Corrective Edits\n(Cyan: +Flood, Magenta: -FalseFlood)", fontsize=11, fontweight='bold')
        axes[2, 3].axis('off')
        
        # Save figure
        fig_save_path = out_fig_dir / f"xai_diagnostic_{scene_id}.png"
        plt.savefig(fig_save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved figure to: {fig_save_path}")
        
        # -------------------------------------------------------------
        # Generate Error Comparison Figure (E0 vs E2 Errors)
        # -------------------------------------------------------------
        fig_err, axes_err = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.08)
        fig_err.suptitle(
            f"Error Comparison: Baseline UNetMamba (E0) vs CAVE-Flood (E2) on {scene_id}\n"
            f"Green=TP, Red=FP, Orange=FN, Dark Gray=TN",
            fontsize=14, fontweight='bold'
        )
        
        axes_err[0].imshow(err_e0)
        axes_err[0].set_title(f"Baseline E0 Errors\n(FP: {fp0:,}, FN: {fn0:,})", fontsize=12, fontweight='bold')
        axes_err[0].axis('off')
        
        axes_err[1].imshow(err_e2)
        axes_err[1].set_title(f"E2 CASR Errors\n(FP: {fp2:,}, FN: {fn2:,})", fontsize=12, fontweight='bold')
        axes_err[1].axis('off')
        
        axes_err[2].imshow(edit_map)
        axes_err[2].set_title(
            f"Flipped Pixels\n(Cyan: {flipped_to_flood.sum():,} Added, Magenta: {flipped_to_bg.sum():,} Removed)",
            fontsize=12, fontweight='bold'
        )
        axes_err[2].axis('off')
        
        # Legend
        patches = [
            mpatches.Patch(color=[0.1, 0.8, 0.2], label="True Positive (TP)"),
            mpatches.Patch(color=[0.9, 0.2, 0.2], label="False Positive (FP)"),
            mpatches.Patch(color=[1.0, 0.65, 0.0], label="False Negative (FN)"),
            mpatches.Patch(color=[0.0, 0.8, 1.0], label="Flipped to Flood (+TP/-FN)"),
            mpatches.Patch(color=[1.0, 0.2, 0.8], label="Flipped to Background (-FP)")
        ]
        fig_err.legend(handles=patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=11)
        
        err_save_path = out_fig_dir / f"error_comparison_{scene_id}.png"
        plt.savefig(err_save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved error comparison to: {err_save_path}")

    # Save summary report JSON
    summary_path = Path("artifacts/CAVE_Flood_Antigravity/stage6_xai_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved XAI summary report to: {summary_path}")
    print("=" * 70)
    print("[STAGE 6 COMPLETE] All qualitative and XAI diagnostics successfully generated.")
    print("=" * 70)

if __name__ == "__main__":
    run_stage6()
