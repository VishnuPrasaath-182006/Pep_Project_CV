"""
CAVE-Flood Inference Engine
Provides cached model management and high-performance forward execution for:
- E0: Frozen UNet-Mamba Baseline
- E1: UNet-Mamba + CASR Adapter (Neutral Guidance)
- E2: UNet-Mamba + CASR Adapter + Frozen RemoteCLIP Vision-Language Guidance
Strictly maintains physical model mathematics, candidate-region gating, and zero-residual invariance.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

DEPLOYMENT_DIR = Path(__file__).resolve().parent
SRC_DIR = DEPLOYMENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deployment_config import (
    MODEL_CONFIG,
    CHECKPOINTS_DIR,
    PATCH_SIZE,
    REGION_SIZE,
    TOP_K,
    MIN_COVERAGE,
    LAMBDA_RES,
    get_device
)
from preprocessing import normalize_image, partition_into_patches
from model import build_model
from casr import CASRAdapter
from candidate_selector import select_candidate_regions, RemoteCLIPGuidance


class ModelManager:
    """
    Singleton cache manager ensuring model weights are loaded once into memory
    and safely reused across multiple user inference queries.
    """
    _instance = None
    
    def __init__(self):
        self.device = get_device()
        self.base_model = None
        self.e1_adapter = None
        self.e2_adapter = None
        self.remoteclip = None
        self.remoteclip_available = False
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def load_base_model(self):
        if self.base_model is None:
            ckpt_path = CHECKPOINTS_DIR / "best_model.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Base UNet-Mamba checkpoint missing: {ckpt_path}")
            model = build_model(MODEL_CONFIG).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.base_model = model
        return self.base_model

    def load_e1_adapter(self):
        if self.e1_adapter is None:
            ckpt_path = CHECKPOINTS_DIR / "best_e1_model.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"CASR E1 checkpoint missing: {ckpt_path}")
            adapter = CASRAdapter(in_channels=9, hidden_channels=32, num_classes=2, lambda_res=LAMBDA_RES).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            adapter.load_state_dict(ckpt["casr_state_dict"])
            adapter.eval()
            for p in adapter.parameters():
                p.requires_grad = False
            self.e1_adapter = adapter
        return self.e1_adapter

    def load_e2_adapter(self):
        if self.e2_adapter is None:
            ckpt_path = CHECKPOINTS_DIR / "best_e2_model.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"CASR E2 checkpoint missing: {ckpt_path}")
            adapter = CASRAdapter(in_channels=9, hidden_channels=32, num_classes=2, lambda_res=LAMBDA_RES).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            adapter.load_state_dict(ckpt["casr_state_dict"])
            adapter.eval()
            for p in adapter.parameters():
                p.requires_grad = False
            self.e2_adapter = adapter
        return self.e2_adapter

    def load_remoteclip(self):
        if self.remoteclip is None:
            scaler_path = CHECKPOINTS_DIR / "remoteclip_score_scaler.json"
            try:
                self.remoteclip = RemoteCLIPGuidance(device=str(self.device), scaler_path=scaler_path)
                self.remoteclip_available = True
            except Exception as e:
                self.remoteclip = None
                self.remoteclip_available = False
                print(f"[RemoteCLIP Warning] Unable to initialize RemoteCLIP: {e}")
        return self.remoteclip


def predict_scene(image_array, valid_mask, mode="E2", run_xai=False):
    """
    Executes full-scene flood segmentation and optional XAI attribution.
    
    Args:
        image_array: np.ndarray (4, H, W) raw PlanetScope uint16 rasters
        valid_mask: np.ndarray (H, W) bool
        mode: str in ["E0", "E1", "E2"]
        run_xai: bool (computes LayerCAM attribution and diagnostic maps if True)
        
    Returns:
        dict of output maps and summary metrics
    """
    mgr = ModelManager.get_instance()
    device = mgr.device
    base_model = mgr.load_base_model()
    
    # Check model mode availability
    if mode == "E2":
        rc = mgr.load_remoteclip()
        if rc is None:
            print("[Fallback Alert] RemoteCLIP weights unavailable; falling back to E1 neutral CASR refinement.")
            mode = "E1"
            
    if mode == "E1":
        adapter = mgr.load_e1_adapter()
    elif mode == "E2":
        adapter = mgr.load_e2_adapter()
    else:
        adapter = None
        
    c, h, w = image_array.shape
    patches, (orig_h, orig_w) = partition_into_patches(image_array, valid_mask, patch_size=PATCH_SIZE)
    
    # Full-scene accumulation arrays
    z_base_full = np.zeros((2, orig_h, orig_w), dtype=np.float32)
    z_final_full = np.zeros((2, orig_h, orig_w), dtype=np.float32)
    cand_mask_full = np.zeros((orig_h, orig_w), dtype=np.float32)
    entropy_full = np.zeros((orig_h, orig_w), dtype=np.float32)
    grad_full = np.zeros((orig_h, orig_w), dtype=np.float32)
    gate_full = np.zeros((orig_h, orig_w), dtype=np.float32)
    guidance_full = np.zeros((orig_h, orig_w), dtype=np.float32)
    layercam_full = np.zeros((orig_h, orig_w), dtype=np.float32) if run_xai else None
    
    # Process each 512x512 patch
    for patch in patches:
        top = patch["top"]
        left = patch["left"]
        ha = patch["h_actual"]
        wa = patch["w_actual"]
        p_raw = patch["image"]
        p_valid = patch["valid_mask"]
        
        # 1. Normalization
        p_norm = normalize_image(p_raw)
        p_tensor = torch.from_numpy(p_norm).unsqueeze(0).float().to(device)
        
        # 2. Baseline UNet-Mamba Pass
        with torch.no_grad():
            z_b = base_model(p_tensor)
            if isinstance(z_b, (tuple, list)):
                z_b = z_b[0]
                
        z_b_np = z_b.squeeze(0).cpu().numpy()
        p_base = np.exp(z_b_np[1] - np.logaddexp(z_b_np[0], z_b_np[1]))
        
        z_base_full[:, top:top+ha, left:left+wa] = z_b_np[:, :ha, :wa]
        
        if mode == "E0":
            z_final_full[:, top:top+ha, left:left+wa] = z_b_np[:, :ha, :wa]
            continue
            
        # 3. Candidate Region Selection (16 regions in 4x4 grid, top 4 selected)
        cand_res = select_candidate_regions(
            flood_prob=p_base,
            valid_mask=p_valid,
            patch_size=PATCH_SIZE,
            region_size=REGION_SIZE,
            top_k=TOP_K,
            min_coverage=MIN_COVERAGE
        )
        cand_mask = cand_res["candidate_mask"]
        selected_regions = cand_res["selected_regions"]
        entropy_map = cand_res["entropy_map"]
        grad_map = cand_res["gradient_map"]
        
        cand_mask_full[top:top+ha, left:left+wa] = cand_mask[:ha, :wa]
        entropy_full[top:top+ha, left:left+wa] = entropy_map[:ha, :wa]
        grad_full[top:top+ha, left:left+wa] = grad_map[:ha, :wa]
        
        # 4. Guidance Channel
        if mode == "E2":
            # RemoteCLIP text-image cosine margin on selected regions
            guidance_map, _ = mgr.remoteclip.build_guidance_map(p_raw, selected_regions)
            guidance_full[top:top+ha, left:left+wa] = guidance_map[:ha, :wa]
        else:
            guidance_map = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            
        # 5. CASR Feature Assembly
        feats = np.concatenate([
            p_norm,
            p_base[np.newaxis, ...],
            entropy_map[np.newaxis, ...],
            grad_map[np.newaxis, ...],
            cand_mask[np.newaxis, ...],
            guidance_map[np.newaxis, ...]
        ], axis=0).astype(np.float32)
        
        feat_t = torch.from_numpy(feats).unsqueeze(0).float().to(device)
        cand_t = torch.from_numpy(cand_mask).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            delta_z, gate, res_update = adapter(feat_t, cand_t)
            z_fin = z_b + res_update
            
        z_fin_np = z_fin.squeeze(0).cpu().numpy()
        gate_np = gate.squeeze().cpu().numpy()
        
        z_final_full[:, top:top+ha, left:left+wa] = z_fin_np[:, :ha, :wa]
        gate_full[top:top+ha, left:left+wa] = gate_np[:ha, :wa]
        
        # 6. Optional LayerCAM Attribution
        if run_xai and adapter is not None:
            from xai_utils import compute_layercam
            cam_patch = compute_layercam(adapter, feat_t, cand_t)
            layercam_full[top:top+ha, left:left+wa] = cam_patch[:ha, :wa]
            
    # Final Discrete Predictions
    pred_base = np.argmax(z_base_full, axis=0)
    pred_final = np.argmax(z_final_full, axis=0)
    
    # Mask invalid pixels
    pred_base[~valid_mask] = 0
    pred_final[~valid_mask] = 0
    
    # Calculate Quantitative Statistics
    total_valid = int(valid_mask.sum())
    base_flood_px = int(((pred_base == 1) & valid_mask).sum())
    refined_flood_px = int(((pred_final == 1) & valid_mask).sum())
    
    base_flood_pct = (base_flood_px / total_valid * 100.0) if total_valid > 0 else 0.0
    refined_flood_pct = (refined_flood_px / total_valid * 100.0) if total_valid > 0 else 0.0
    
    # Pixel edit transitions
    added_flood_px = int(((pred_base == 0) & (pred_final == 1) & valid_mask).sum())
    suppressed_px = int(((pred_base == 1) & (pred_final == 0) & valid_mask).sum())
    
    stats = {
        "mode": mode,
        "total_valid_pixels": total_valid,
        "base_flood_pixels": base_flood_px,
        "base_flood_pct": round(base_flood_pct, 2),
        "refined_flood_pixels": refined_flood_px,
        "refined_flood_pct": round(refined_flood_pct, 2),
        "added_flood_pixels": added_flood_px,
        "suppressed_pixels": suppressed_px,
        "candidate_coverage_pct": round(((cand_mask_full > 0) & valid_mask).sum() / total_valid * 100.0, 2) if total_valid > 0 else 0.0
    }
    
    # Residual difference probability map
    p_base_full = np.exp(z_base_full[1] - np.logaddexp(z_base_full[0], z_base_full[1]))
    p_final_full = np.exp(z_final_full[1] - np.logaddexp(z_final_full[0], z_final_full[1]))
    prob_delta = (p_final_full - p_base_full) * valid_mask.astype(np.float32)
    
    return {
        "pred_base": pred_base,
        "pred_final": pred_final,
        "p_base": p_base_full,
        "p_final": p_final_full,
        "prob_delta": prob_delta,
        "cand_mask": cand_mask_full,
        "entropy_map": entropy_full,
        "gradient_map": grad_full,
        "gate_map": gate_full,
        "guidance_map": guidance_full,
        "layercam_map": layercam_full,
        "stats": stats
    }
