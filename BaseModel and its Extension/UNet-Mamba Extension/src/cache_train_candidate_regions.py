"""
Cache candidate regions and 9-channel features for training patches across all 14 training events.
Ensures E1 and E2 have identical candidate masks, features, labels, and order.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import load_config, BASE_DIR, TRAIN_EVENTS
from model import build_model
from candidate_selector import (
    select_candidate_regions,
    RemoteCLIPGuidance
)


def cache_training_patches(
    config=None,
    num_scenes_per_event=2,
    save_dir=None
):
    if config is None:
        config = load_config()
        
    device = torch.device("cpu")
    model = build_model(config).to(device)
    model.eval()
    
    ckpt_path = BASE_DIR / "checkpoints" / "best_model.pth"
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
    
    scaler_path = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "remoteclip_score_scaler.json"
    rc = RemoteCLIPGuidance(device="cpu", scaler_path=scaler_path)
    
    if save_dir is None:
        save_dir = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "cache" / "candidate_regions" / "train"
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    train_manifest = BASE_DIR / "manifests" / "train.csv"
    df_train_scenes = pd.read_csv(train_manifest)
    train_patches_dir = BASE_DIR / "manifests" / "processed_data" / "processed_data" / "train"
    
    norm_mean = np.array(config["data"]["norm_mean"], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config["data"]["norm_std"], dtype=np.float32).reshape(4, 1, 1)
    
    print("=" * 65)
    print("PREPARING TRAINING CANDIDATE REGIONS ACROSS ALL 14 EVENTS")
    print("=" * 65)
    
    cached_count = 0
    records = []
    
    for event_name in TRAIN_EVENTS:
        df_event = df_train_scenes[df_train_scenes["event"] == event_name]
        sampled_stems = df_event["stem"].iloc[:num_scenes_per_event].tolist()
        
        for stem in sampled_stems:
            for p_idx, top, left in [(0, 0, 0), (1, 0, 512), (2, 512, 0), (3, 512, 512)]:
                patch_name = f"{event_name}_{stem}_p{p_idx}_{top}_{left}.npz"
                p_file = train_patches_dir / patch_name
                if not p_file.exists():
                    continue
                    
                out_file = save_dir / f"{event_name}_{stem}_p{p_idx}.npz"
                if out_file.exists():
                    cached_count += 1
                    records.append(str(out_file))
                    continue
                    
                p_data = np.load(p_file)
                img = p_data["image"]
                mask = p_data["mask"]
                
                # Baseline model inference
                img_norm = (img.astype(np.float32) - norm_mean) / (norm_std + 1e-6)
                img_t = torch.from_numpy(img_norm).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    logits = model(img_t)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                    patch_logits = logits[0].cpu().numpy()
                    
                valid_mask = (img != 0).any(axis=0) & (mask != -1)
                
                # Candidate selection
                cand_res = select_candidate_regions(probs, valid_mask=valid_mask)
                cand_mask = cand_res["candidate_mask"]
                selected_regions = cand_res["selected_regions"]
                entropy_map = cand_res["entropy_map"]
                grad_map = cand_res["gradient_map"]
                
                # E1: Neutral (zeros)
                guidance_e1 = np.zeros((512, 512), dtype=np.float32)
                # E2: RemoteCLIP scaled margin
                guidance_e2, region_margins = rc.build_guidance_map(img, selected_regions)
                
                np.savez_compressed(
                    out_file,
                    candidate_mask=cand_mask.astype(np.float32),
                    guidance_e1=guidance_e1.astype(np.float32),
                    guidance_e2=guidance_e2.astype(np.float32),
                    entropy_map=entropy_map.astype(np.float32),
                    gradient_map=grad_map.astype(np.float32),
                    flood_prob=probs.astype(np.float32),
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
                cached_count += 1
                records.append(str(out_file))
                
        print(f"  Cached event: {event_name:12s} -> Total patches: {cached_count}")
        
    manifest_file = save_dir / "train_cached_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    print(f"\nCompleted! Cached {len(records)} training patches in: {save_dir}")
    return records


if __name__ == "__main__":
    cache_training_patches()
