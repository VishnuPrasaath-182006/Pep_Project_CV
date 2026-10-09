"""
CAVE-Flood Demonstration Inference Pipeline
Provides end-to-end inference on 4-band PlanetScope imagery:
- Accepts raw 4-band array or patch
- Produces baseline flood mask (E0)
- Produces refined flood mask (E1 or E2)
- Produces prediction overlays, flood area percentage, and uncertainty maps
- Allows CASR to be disabled via --disable_casr switch
"""

import sys
import argparse
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

from config import load_config, BASE_DIR
from model import build_model
from casr import CASRAdapter
from candidate_selector import select_candidate_regions, RemoteCLIPGuidance


def run_demo_inference(
    patch_path,
    output_dir="results/demo_output",
    mode="E2",
    disable_casr=False
):
    patch_path = Path(patch_path)
    assert patch_path.exists(), f"Input patch not found: {patch_path}"
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config()
    device = torch.device("cpu")
    
    # Load input patch
    data = np.load(patch_path)
    if "image" in data:
        raw_img = data["image"]
        raw_mask = data.get("mask", None)
    elif "raw_image" in data:
        raw_img = data["raw_image"]
        raw_mask = data.get("raw_mask", None)
    else:
        raise ValueError(f"Unrecognized keys in {patch_path}: {list(data.keys())}")
        
    h, w = raw_img.shape[1], raw_img.shape[2]
    
    # Normalization
    norm_mean = np.array(config["data"]["norm_mean"], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config["data"]["norm_std"], dtype=np.float32).reshape(4, 1, 1)
    img_norm = (raw_img.astype(np.float32) - norm_mean) / (norm_std + 1e-6)
    
    # 1. Base UNet-Mamba Model
    base_model = build_model(config).to(device)
    ckpt_path = BASE_DIR / "checkpoints" / "best_model.pth"
    base_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()
    
    # Baseline forward pass
    x_t = torch.from_numpy(img_norm).unsqueeze(0).float().to(device)
    with torch.no_grad():
        z_base = base_model(x_t)
        if isinstance(z_base, (tuple, list)):
            z_base = z_base[0]
            
    z_base_np = z_base.squeeze(0).cpu().numpy()
    p_base = np.exp(z_base_np[1] - np.logaddexp(z_base_np[0], z_base_np[1]))
    pred_base = np.argmax(z_base_np, axis=0)
    
    # Validity mask
    valid_mask = (raw_img != 0).any(axis=0)
    if raw_mask is not None:
        valid_mask = valid_mask & (raw_mask != -1)
    total_valid_pixels = int(valid_mask.sum())
    
    base_flood_pixels = int(((pred_base == 1) & valid_mask).sum())
    base_flood_pct = (base_flood_pixels / total_valid_pixels * 100.0) if total_valid_pixels > 0 else 0.0
    
    print("=" * 60)
    print(f"CAVE-FLOOD DEMO INFERENCE: {patch_path.name}")
    print("=" * 60)
    print(f"Input Shape: {raw_img.shape} (Channels: 4 PlanetScope B, G, R, NIR)")
    print(f"Total Valid Pixels: {total_valid_pixels:,} / {h*w:,}")
    print(f"Baseline (E0) Flood Area: {base_flood_pct:.2f}% ({base_flood_pixels:,} pixels)")
    
    if disable_casr:
        print("[CASR BYPASS ACTIVATED] Returning baseline prediction only.")
        pred_final = pred_base
        z_final_np = z_base_np
        refined_flood_pct = base_flood_pct
        gate_np = np.zeros((h, w), dtype=np.float32)
        cand_mask = np.zeros((h, w), dtype=np.float32)
    else:
        # 2. Candidate Region Selection
        cand_res = select_candidate_regions(
            flood_prob=p_base,
            valid_mask=valid_mask,
            patch_size=512,
            region_size=128,
            top_k=4,
            min_coverage=0.5
        )
        cand_mask = cand_res["candidate_mask"]
        selected_regions = cand_res["selected_regions"]
        entropy_map = cand_res["entropy_map"]
        grad_map = cand_res["gradient_map"]
        
        # 3. Guidance Channel
        if mode == "E2":
            scaler_path = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "remoteclip_score_scaler.json"
            rc_guidance = RemoteCLIPGuidance(device="cpu", scaler_path=scaler_path)
            guidance_map, _ = rc_guidance.build_guidance_map(raw_img, selected_regions)
            adapter_ckpt = BASE_DIR / "checkpoints" / "best_e2_model.pth"
        else:
            guidance_map = np.zeros((h, w), dtype=np.float32)
            adapter_ckpt = BASE_DIR / "checkpoints" / "best_e1_model.pth"
            
        # 4. CASR Adapter
        adapter = CASRAdapter(in_channels=9, hidden_channels=32, num_classes=2, lambda_res=1.0)
        ad_data = torch.load(adapter_ckpt, map_location=device, weights_only=True)
        adapter.load_state_dict(ad_data["casr_state_dict"])
        adapter.eval()
        
        # Features
        feats = np.concatenate([
            img_norm,
            p_base[np.newaxis, ...],
            entropy_map[np.newaxis, ...],
            grad_map[np.newaxis, ...],
            cand_mask[np.newaxis, ...],
            guidance_map[np.newaxis, ...]
        ], axis=0).astype(np.float32)
        
        feat_t = torch.from_numpy(feats).unsqueeze(0).float()
        cand_t = torch.from_numpy(cand_mask).unsqueeze(0).unsqueeze(0).float()
        
        with torch.no_grad():
            delta_z, gate, res_update = adapter(feat_t, cand_t)
            z_final = z_base + res_update
            
        z_final_np = z_final.squeeze(0).numpy()
        gate_np = gate.squeeze().numpy()
        pred_final = np.argmax(z_final_np, axis=0)
        
        refined_flood_pixels = int(((pred_final == 1) & valid_mask).sum())
        refined_flood_pct = (refined_flood_pixels / total_valid_pixels * 100.0) if total_valid_pixels > 0 else 0.0
        
        flipped_to_flood = int(((pred_base == 0) & (pred_final == 1) & valid_mask).sum())
        flipped_to_bg = int(((pred_base == 1) & (pred_final == 0) & valid_mask).sum())
        
        print(f"Refined ({mode}) Flood Area: {refined_flood_pct:.2f}% ({refined_flood_pixels:,} pixels)")
        print(f"CASR Pixel Edits: +{flipped_to_flood:,} flood pixels added | -{flipped_to_bg:,} false flood pixels suppressed")
        
    # Visualization plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. RGB
    rgb_raw = raw_img[:3].transpose(1, 2, 0).astype(np.float32)
    p2, p98 = np.percentile(rgb_raw[valid_mask], (2, 98), axis=0) if valid_mask.sum() > 0 else (0, 1)
    rgb_disp = np.clip((rgb_raw - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    rgb_disp[~valid_mask] = 0.0
    axes[0].imshow(rgb_disp)
    axes[0].set_title("PlanetScope RGB Imagery")
    axes[0].axis('off')
    
    # 2. Baseline Mask
    m0_disp = np.ma.masked_where(~valid_mask, pred_base)
    axes[1].imshow(m0_disp, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_facecolor('black')
    axes[1].set_title(f"Baseline Flood Mask ({base_flood_pct:.1f}%)")
    axes[1].axis('off')
    
    # 3. Refined Mask
    m_disp = np.ma.masked_where(~valid_mask, pred_final)
    axes[2].imshow(m_disp, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_facecolor('black')
    axes[2].set_title(f"Refined {mode} Flood Mask ({refined_flood_pct:.1f}%)")
    axes[2].axis('off')
    
    # 4. Difference Map
    diff_map = np.zeros((h, w, 3), dtype=np.float32)
    diff_map[(pred_base == pred_final) & valid_mask] = [0.15, 0.15, 0.2]
    diff_map[(pred_base == 0) & (pred_final == 1) & valid_mask] = [0.0, 0.8, 1.0] # Cyan: added flood
    diff_map[(pred_base == 1) & (pred_final == 0) & valid_mask] = [1.0, 0.2, 0.8] # Magenta: removed flood
    axes[3].imshow(diff_map)
    axes[3].set_title("CASR Edits (Cyan:+Water, Magenta:-Water)")
    axes[3].axis('off')
    
    save_fig_path = out_dir / f"demo_inference_{patch_path.stem}.png"
    plt.savefig(save_fig_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_fig_path}")
    print("=" * 60)
    return {
        "base_flood_pct": base_flood_pct,
        "refined_flood_pct": refined_flood_pct,
        "fig_path": str(save_fig_path)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_path", type=str, default="artifacts/CAVE_Flood_Antigravity/cache/candidate_regions/val/Bolivia_BOL_1015_p0.npz")
    parser.add_argument("--mode", type=str, default="E2", choices=["E1", "E2"])
    parser.add_argument("--disable_casr", action="store_true")
    args = parser.parse_args()
    run_demo_inference(args.patch_path, mode=args.mode, disable_casr=args.disable_casr)
