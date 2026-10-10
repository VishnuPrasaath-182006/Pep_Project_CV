"""
Independent Ground-Truth Pixel Counter for Held-Out Test Partition
Reads test labels directly without model inference and counts:
- Total valid pixels
- Ground-truth flood pixels (label == 1)
- Ground-truth background pixels (label == 0)
- NoData pixels (image == 0)
- Ignored pixels (label == -1)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def count_gt_pixels():
    src_dir = Path(__file__).resolve().parent
    base_dir = src_dir.parent
    manifest_path = Path("UNet-Mamba/manifests/test.csv")
    if not manifest_path.exists():
        manifest_path = base_dir / "manifests" / "test.csv"
    patches_dir = Path("UNet-Mamba/manifests/processed_data/processed_data/test")
    if not patches_dir.exists():
        patches_dir = base_dir / "manifests" / "processed_data" / "processed_data" / "test"
    
    df_test = pd.read_csv(manifest_path)
    quadrant_specs = [
        (0, 0, "p0_0_0"),
        (0, 512, "p1_0_512"),
        (512, 0, "p2_512_0"),
        (512, 512, "p3_512_512")
    ]
    
    event_stats = {}
    
    for idx in range(len(df_test)):
        row = df_test.iloc[idx]
        evt = row["event"]
        stem = row["stem"]
        
        if evt not in event_stats:
            event_stats[evt] = {
                "scenes": 0,
                "valid": 0,
                "flood": 0,
                "bg": 0,
                "nodata": 0,
                "ignore": 0
            }
        event_stats[evt]["scenes"] += 1
        
        recon_img = np.zeros((4, 1024, 1024), dtype=np.uint16)
        recon_mask = np.full((1024, 1024), -1, dtype=np.int64)
        
        for top, left, q_tag in quadrant_specs:
            p_file = patches_dir / f"{evt}_{stem}_{q_tag}.npz"
            p_data = np.load(p_file)
            recon_img[:, top:top+512, left:left+512] = p_data["image"]
            recon_mask[top:top+512, left:left+512] = p_data["mask"]
            
        valid = (recon_img != 0).any(axis=0) & (recon_mask != -1)
        
        event_stats[evt]["valid"] += int(valid.sum())
        event_stats[evt]["flood"] += int(((recon_mask == 1) & valid).sum())
        event_stats[evt]["bg"] += int(((recon_mask == 0) & valid).sum())
        event_stats[evt]["nodata"] += int(((recon_img == 0).all(axis=0)).sum())
        event_stats[evt]["ignore"] += int((recon_mask == -1).sum())

    print("=" * 80)
    print("INDEPENDENT GROUND-TRUTH TEST LABEL AUDIT (NO MODEL INFERENCE)")
    print("=" * 80)
    
    total_scenes = 0
    total_valid = 0
    total_flood = 0
    total_bg = 0
    total_nodata = 0
    total_ignore = 0
    
    for evt, s in event_stats.items():
        print(f"Event: {evt:<12s} | Scenes: {s['scenes']:2d} | Valid: {s['valid']:10,d} | Flood GT: {s['flood']:10,d} | BG GT: {s['bg']:10,d} | NoData: {s['nodata']:4d} | Ignore: {s['ignore']:4d}")
        total_scenes += s["scenes"]
        total_valid += s["valid"]
        total_flood += s["flood"]
        total_bg += s["bg"]
        total_nodata += s["nodata"]
        total_ignore += s["ignore"]
        
    print("-" * 80)
    print(f"TOTAL HELD-OUT TEST : Scenes: {total_scenes:2d} | Valid: {total_valid:10,d} | Flood GT: {total_flood:10,d} | BG GT: {total_bg:10,d} | NoData: {total_nodata:4d} | Ignore: {total_ignore:4d}")
    print("=" * 80)
    
    # Comparison against reports:
    print("\nCOMPARISON AGAINST REPORTS:")
    print("1. Original Report (v1):")
    v1_flood = 11211885 + 2204591  # TP + FN
    v1_bg = 31687383 + 1033485     # TN + FP
    print(f"   Published v1 Flood GT (TP+FN): {v1_flood:10,d}  [MISMATCH with true labels: delta = {v1_flood - total_flood:+,d}]")
    print(f"   Published v1 BG GT    (TN+FP): {v1_bg:10,d}  [MISMATCH with true labels: delta = {v1_bg - total_bg:+,d}]")
    
    print("\n2. Evaluator Ground-Truth / v2 Report:")
    v2_flood = 22201266 + 4365354  # TP + FN
    v2_bg = 17524242 + 2046482     # TN + FP
    print(f"   Evaluator/v2 Flood GT (TP+FN): {v2_flood:10,d}  [MATCHES TRUE LABELS EXACTLY: delta = {v2_flood - total_flood:d}]")
    print(f"   Evaluator/v2 BG GT    (TN+FP): {v2_bg:10,d}  [MATCHES TRUE LABELS EXACTLY: delta = {v2_bg - total_bg:d}]")
    
    assert total_valid == 46137344
    assert total_flood == 26566620
    assert total_bg == 19570724
    assert v2_flood == total_flood
    print("\nALL GROUND-TRUTH VERIFICATION ASSERTIONS PASSED IDENTICALLY.")
    return {
        "valid_pixels": total_valid,
        "flood_pixels": total_flood,
        "bg_pixels": total_bg,
        "nodata_pixels": total_nodata,
        "ignore_pixels": total_ignore,
        "event_stats": event_stats
    }

if __name__ == "__main__":
    count_gt_pixels()
