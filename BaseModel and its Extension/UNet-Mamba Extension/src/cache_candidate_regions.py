"""
Candidate-Region Selection & RemoteCLIP Guidance Caching Script (Stage 3)
1. Fits RemoteCLIP score scaler using candidate regions from training events.
2. Caches candidate masks, guidance maps (E1 neutral, E2 RemoteCLIP), entropy,
   and gradient maps for validation patches.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import load_config, BASE_DIR, TRAIN_EVENTS, VAL_EVENTS
from model import build_model
from candidate_selector import (
    select_candidate_regions,
    compute_entropy_and_gradient,
    RemoteCLIPGuidance
)


def fit_training_scaler(
    config=None,
    num_samples_per_event=3,
    save_path=None
):
    if config is None:
        config = load_config()
        
    device = torch.device("cpu")
    model = build_model(config).to(device)
    model.eval()
    
    ckpt_path = BASE_DIR / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        zip_fallback = ckpt_path.with_suffix(".pth.zip")
        if zip_fallback.exists():
            import shutil
            shutil.copyfile(zip_fallback, ckpt_path)
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
    
    rc = RemoteCLIPGuidance(device="cpu")
    
    train_manifest = BASE_DIR / "manifests" / "train.csv"
    df_train_scenes = pd.read_csv(train_manifest)
    train_patches_dir = BASE_DIR / "manifests" / "processed_data" / "processed_data" / "train"
    
    norm_mean = np.array(config["data"]["norm_mean"], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config["data"]["norm_std"], dtype=np.float32).reshape(4, 1, 1)
    
    print("=" * 65)
    print("STAGE 3: FITTING REMOTECLIP SCORE SCALER ON TRAINING EVENTS")
    print("=" * 65)
    print(f"Sampling {num_samples_per_event} scenes from each of the 14 training events...")
    
    raw_margins = []
    
    for event_name in TRAIN_EVENTS:
        df_event = df_train_scenes[df_train_scenes["event"] == event_name]
        sampled_stems = df_event["stem"].iloc[:num_samples_per_event].tolist()
        
        for stem in sampled_stems:
            # Quadrants p0, p1, p2, p3
            for p_idx, top, left in [(0, 0, 0), (1, 0, 512), (2, 512, 0), (3, 512, 512)]:
                p_file = train_patches_dir / f"{event_name}_{stem}_p{p_idx}_{top}_{left}.npz"
                if not p_file.exists():
                    continue
                p_data = np.load(p_file)
                img = p_data["image"]
                mask = p_data["mask"]
                
                # Model inference for flood prob
                img_norm = (img.astype(np.float32) - norm_mean) / (norm_std + 1e-6)
                img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1)[0, 1].numpy()
                    
                valid_mask = (img != 0).any(axis=0) & (mask != -1)
                
                # Candidate selection
                res = select_candidate_regions(probs, valid_mask=valid_mask)
                selected = res["selected_regions"]
                
                # RemoteCLIP raw margins
                for reg in selected:
                    r_top = reg["top"]
                    r_left = reg["left"]
                    reg_rgb_raw = img[:3, r_top:r_top+128, r_left:r_left+128].astype(np.float32)
                    
                    reg_rgb_uint8 = np.zeros((128, 128, 3), dtype=np.uint8)
                    for c in range(3):
                        c_min = np.percentile(reg_rgb_raw[c], 1)
                        c_max = np.percentile(reg_rgb_raw[c], 99)
                        if c_max > c_min:
                            norm_c = np.clip((reg_rgb_raw[c] - c_min) / (c_max - c_min), 0.0, 1.0)
                        else:
                            norm_c = np.zeros_like(reg_rgb_raw[c])
                        reg_rgb_uint8[:, :, c] = (norm_c * 255.0).astype(np.uint8)
                        
                    raw_margin = rc.compute_region_margin(reg_rgb_uint8)
                    raw_margins.append(raw_margin)
                    
        print(f"  Event: {event_name:12s} -> Total accumulated regions: {len(raw_margins)}")
        
    if save_path is None:
        save_path = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "remoteclip_score_scaler.json"
        
    scaler_info = rc.fit_scaler(raw_margins, save_path=save_path)
    return rc, scaler_info


def cache_validation_candidate_regions(
    config=None,
    rc_guidance=None,
    scaler_path=None,
    split="val",
    cache_dir=None
):
    if config is None:
        config = load_config()
        
    if scaler_path is None:
        scaler_path = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "remoteclip_score_scaler.json"
        
    if rc_guidance is None:
        rc_guidance = RemoteCLIPGuidance(device="cpu", scaler_path=scaler_path)
        
    if cache_dir is None:
        cache_dir = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "cache" / "candidate_regions" / split
    cache_dir = Path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    val_manifest = BASE_DIR / "manifests" / f"{split}.csv"
    df_scenes = pd.read_csv(val_manifest)
    val_patches_dir = BASE_DIR / "manifests" / "processed_data" / "processed_data" / split
    e0_logits_dir = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "cache" / f"{split}_e0_logits"
    
    print("\n" + "=" * 65)
    print(f"CACHING CANDIDATE REGIONS & GUIDANCE FOR {split.upper()} SPLIT")
    print("=" * 65)
    
    total_patches = 0
    total_regions_selected = 0
    
    for s_idx in tqdm(range(len(df_scenes)), desc=f"Caching {split} candidate data"):
        row = df_scenes.iloc[s_idx]
        event_name = row["event"]
        stem = row["stem"]
        
        # Load cached 1024x1024 baseline logits
        logit_file = e0_logits_dir / f"{event_name}_{stem}_logits.npz"
        if not logit_file.exists():
            raise FileNotFoundError(f"Missing cached baseline logits: {logit_file}. Run Stage 2 first!")
        chip_logits = np.load(logit_file)["logits"].astype(np.float32)  # (2, 1024, 1024)
        
        quadrants = [
            (0, 0, 0, f"{event_name}_{stem}_p0_0_0.npz"),
            (1, 0, 512, f"{event_name}_{stem}_p1_0_512.npz"),
            (2, 512, 0, f"{event_name}_{stem}_p2_512_0.npz"),
            (3, 512, 512, f"{event_name}_{stem}_p3_512_512.npz")
        ]
        
        for p_idx, top, left, patch_name in quadrants:
            out_file = cache_dir / f"{event_name}_{stem}_p{p_idx}.npz"
            if out_file.exists():
                total_patches += 1
                continue
                
            p_file = val_patches_dir / patch_name
            p_data = np.load(p_file)
            img = p_data["image"]
            mask = p_data["mask"]
            
            # Slice patch logits (2, 512, 512)
            patch_logits = chip_logits[:, top:top+512, left:left+512]
            
            # Compute baseline flood probability (softmax)
            exp_l = np.exp(patch_logits - np.max(patch_logits, axis=0, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=0, keepdims=True)
            flood_prob = probs[1]
            
            valid_mask = (img != 0).any(axis=0) & (mask != -1)
            
            # Select Candidate Regions
            cand_res = select_candidate_regions(flood_prob, valid_mask=valid_mask)
            cand_mask = cand_res["candidate_mask"]  # (512, 512)
            selected_regions = cand_res["selected_regions"]
            entropy_map = cand_res["entropy_map"]
            grad_map = cand_res["gradient_map"]
            
            # Build Guidance Maps
            # E1: Identically zero
            guidance_e1 = np.zeros((512, 512), dtype=np.float32)
            # E2: RemoteCLIP scaled margin
            guidance_e2, region_margins = rc_guidance.build_guidance_map(img, selected_regions)
            
            total_regions_selected += len(selected_regions)
            total_patches += 1
            
            # Save compressed cache
            np.savez_compressed(
                out_file,
                candidate_mask=cand_mask.astype(np.float32),
                guidance_e1=guidance_e1.astype(np.float32),
                guidance_e2=guidance_e2.astype(np.float32),
                entropy_map=entropy_map.astype(np.float32),
                gradient_map=grad_map.astype(np.float32),
                flood_prob=flood_prob.astype(np.float32),
                patch_logits=patch_logits.astype(np.float16),
                raw_image=img,
                raw_mask=mask,
                metadata=json.dumps({
                    "event": event_name,
                    "stem": stem,
                    "patch_idx": p_idx,
                    "top": top,
                    "left": left,
                    "selected_regions": selected_regions,
                    "region_margins": region_margins
                })
            )
            
    print(f"Cached {total_patches} patches in: {cache_dir}")
    print(f"Total candidate regions selected across {split}: {total_regions_selected}")
    return total_patches


if __name__ == "__main__":
    rc, scaler_info = fit_training_scaler(num_samples_per_event=2)
    cache_validation_candidate_regions(rc_guidance=rc, split="val")
