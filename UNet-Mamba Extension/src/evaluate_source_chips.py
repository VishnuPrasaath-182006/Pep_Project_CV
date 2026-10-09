"""
Source-Chip Evaluator: Unique-Source-Pixel Reconstruction and Evaluation
Implements chip reconstruction from 512x512 patches into 1024x1024 source chips,
validity masking (image valid & label non-ignore), resume-safe logit caching,
and rigorous unique-source-pixel metric calculation.
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

from config import load_config, BASE_DIR, VAL_EVENTS, TEST_EVENTS
from model import build_model


def compute_binary_metrics(tp, fp, fn, tn, chip_count=0):
    total_valid = tp + fp + fn + tn
    
    # Flood IoU (Primary metric)
    denom_flood = tp + fp + fn
    flood_iou = float(tp / denom_flood) if denom_flood > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
    
    # Background IoU
    denom_bg = tn + fp + fn
    bg_iou = float(tn / denom_bg) if denom_bg > 0 else 0.0
    
    # Macro Mean IoU
    mean_iou = float((flood_iou + bg_iou) / 2.0)
    
    # Flood Dice / F1
    denom_dice = 2 * tp + fp + fn
    flood_dice = float(2 * tp / denom_dice) if denom_dice > 0 else (1.0 if denom_dice == 0 else 0.0)
    
    # Precision & Recall
    flood_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    flood_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # Specificity
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Pixel Accuracy
    pixel_accuracy = float((tp + tn) / total_valid) if total_valid > 0 else 0.0
    
    return {
        "flood_iou": flood_iou,
        "bg_iou": bg_iou,
        "mean_iou": mean_iou,
        "flood_dice": flood_dice,
        "flood_precision": flood_precision,
        "flood_recall": flood_recall,
        "specificity": specificity,
        "pixel_accuracy": pixel_accuracy,
        "valid_pixels": int(total_valid),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "chip_count": int(chip_count)
    }


def evaluate_source_chips(
    model_or_path=None,
    config=None,
    split="val",
    cache_dir=None,
    output_dir=None,
    max_chips=None,
    experiment_name="E0"
):
    if config is None:
        config = load_config()
        
    device = torch.device("cpu")
    
    # 1. Load Model
    if isinstance(model_or_path, torch.nn.Module):
        model = model_or_path
    else:
        model = build_model(config).to(device)
        model.eval()
        if model_or_path is None:
            model_or_path = BASE_DIR / "checkpoints" / "best_model.pth"
        model_path = Path(model_or_path)
        if not model_path.exists():
            zip_fallback = model_path.with_suffix(".pth.zip")
            if zip_fallback.exists():
                import shutil
                shutil.copyfile(zip_fallback, model_path)
        ckpt_data = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        print(f"[{experiment_name}] Loaded model from: {model_path} (Epoch {ckpt_data.get('epoch')})")
        
    model.eval()
    
    # 2. Setup Directories
    if cache_dir is None:
        cache_dir = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "cache" / f"{split}_{experiment_name.lower()}_logits"
    cache_dir = Path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    if output_dir is None:
        output_dir = BASE_DIR / "results" / "metrics"
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Load Split Manifest
    manifest_csv = BASE_DIR / "manifests" / f"{split}.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    df_scenes = pd.read_csv(manifest_csv)
    
    if max_chips is not None and max_chips > 0:
        df_scenes = df_scenes.iloc[:max_chips]
        print(f"Limiting evaluation to first {max_chips} chips.")
        
    patches_dir = BASE_DIR / "manifests" / "processed_data" / "processed_data" / split
    norm_mean = np.array(config["data"]["norm_mean"], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config["data"]["norm_std"], dtype=np.float32).reshape(4, 1, 1)
    
    # Confusion counts trackers
    pooled_tp = 0
    pooled_fp = 0
    pooled_fn = 0
    pooled_tn = 0
    
    event_counts = {}
    for evt in df_scenes["event"].unique():
        event_counts[evt] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "chips": 0}
        
    print(f"\n=======================================================")
    print(f"[{experiment_name}] EVALUATING {len(df_scenes)} SOURCE CHIPS ({split.upper()} SPLIT)")
    print(f"=======================================================")
    
    start_time = time.time()
    
    for idx in tqdm(range(len(df_scenes)), desc=f"{experiment_name} {split} evaluation"):
        row = df_scenes.iloc[idx]
        event_name = row["event"]
        stem = row["stem"]
        
        cache_file = cache_dir / f"{event_name}_{stem}_logits.npz"
        
        # Quadrant layout
        quadrants = [
            (0, 0, f"{event_name}_{stem}_p0_0_0.npz"),
            (0, 512, f"{event_name}_{stem}_p1_0_512.npz"),
            (512, 0, f"{event_name}_{stem}_p2_512_0.npz"),
            (512, 512, f"{event_name}_{stem}_p3_512_512.npz")
        ]
        
        # Reconstruct 1024x1024 arrays
        recon_img = np.zeros((4, 1024, 1024), dtype=np.uint16)
        recon_mask = np.full((1024, 1024), -1, dtype=np.int64)
        
        for top, left, patch_name in quadrants:
            p_file = patches_dir / patch_name
            if not p_file.exists():
                raise FileNotFoundError(f"Missing patch file: {p_file}")
            p_data = np.load(p_file)
            recon_img[:, top:top+512, left:left+512] = p_data["image"]
            recon_mask[top:top+512, left:left+512] = p_data["mask"]
            
        # Reconstruct or load cached logits
        if cache_file.exists():
            cached_data = np.load(cache_file)
            recon_logits = cached_data["logits"].astype(np.float32)
        else:
            recon_logits = np.zeros((2, 1024, 1024), dtype=np.float32)
            for top, left, patch_name in quadrants:
                patch_img = recon_img[:, top:top+512, left:left+512].astype(np.float32)
                patch_norm = (patch_img - norm_mean) / (norm_std + 1e-6)
                patch_tensor = torch.from_numpy(patch_norm).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    logits_patch = model(patch_tensor)  # (1, 2, 512, 512)
                recon_logits[:, top:top+512, left:left+512] = logits_patch[0].cpu().numpy()
                
            # Save compressed float16 logits
            np.savez_compressed(cache_file, logits=recon_logits.astype(np.float16))
            
        # Validity mask construction
        img_valid = (recon_img != 0).any(axis=0)  # Not all 4 bands are 0
        mask_valid = (recon_mask != -1)           # Valid label (0 or 1)
        valid_pixels = img_valid & mask_valid
        
        # Predictions
        preds = np.argmax(recon_logits, axis=0)  # (1024, 1024)
        
        # Calculate confusion metrics on valid pixels
        y_true = recon_mask[valid_pixels]
        y_pred = preds[valid_pixels]
        
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        
        # Update pooled counts
        pooled_tp += tp
        pooled_fp += fp
        pooled_fn += fn
        pooled_tn += tn
        
        # Update event counts
        event_counts[event_name]["tp"] += tp
        event_counts[event_name]["fp"] += fp
        event_counts[event_name]["fn"] += fn
        event_counts[event_name]["tn"] += tn
        event_counts[event_name]["chips"] += 1
        
    elapsed = time.time() - start_time
    print(f"\nCompleted evaluation in {elapsed:.1f} s ({elapsed / len(df_scenes):.2f} s/chip)")
    
    # 4. Compute Metrics
    pooled_metrics = compute_binary_metrics(pooled_tp, pooled_fp, pooled_fn, pooled_tn, len(df_scenes))
    
    per_event_records = []
    for evt, counts in sorted(event_counts.items()):
        evt_metrics = compute_binary_metrics(
            counts["tp"], counts["fp"], counts["fn"], counts["tn"], counts["chips"]
        )
        evt_metrics["event"] = evt
        per_event_records.append(evt_metrics)
        
    df_per_event = pd.DataFrame(per_event_records)
    cols = ["event", "chip_count", "valid_pixels", "flood_iou", "bg_iou", "mean_iou", "flood_dice", "flood_precision", "flood_recall", "specificity", "pixel_accuracy", "tp", "fp", "fn", "tn"]
    df_per_event = df_per_event[cols]
    
    # Macro averages across events
    macro_event_flood_iou = float(df_per_event["flood_iou"].mean())
    macro_event_mean_iou = float(df_per_event["mean_iou"].mean())
    macro_event_dice = float(df_per_event["flood_dice"].mean())
    
    summary = {
        "experiment": experiment_name,
        "split": split,
        "total_chips": len(df_scenes),
        "total_valid_pixels": pooled_metrics["valid_pixels"],
        "pixel_pooled": pooled_metrics,
        "macro_event_averages": {
            "macro_event_flood_iou": macro_event_flood_iou,
            "macro_event_mean_iou": macro_event_mean_iou,
            "macro_event_dice": macro_event_dice
        },
        "per_event": per_event_records
    }
    
    # 5. Save Outputs
    json_path = output_dir / f"{experiment_name.lower()}_{split}_metrics_unique_source_pixels.json"
    csv_path = output_dir / f"{experiment_name.lower()}_{split}_per_event_metrics.csv"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    df_per_event.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print(f"[{experiment_name} {split.upper()} RESULTS - UNIQUE SOURCE PIXELS]")
    print("=" * 70)
    print(f"Pixel-Pooled Flood IoU:     {pooled_metrics['flood_iou']:.4f} (Primary)")
    print(f"Pixel-Pooled Macro mIoU:    {pooled_metrics['mean_iou']:.4f}")
    print(f"Pixel-Pooled Flood Dice:    {pooled_metrics['flood_dice']:.4f}")
    print(f"Pixel-Pooled Precision:     {pooled_metrics['flood_precision']:.4f}")
    print(f"Pixel-Pooled Recall:        {pooled_metrics['flood_recall']:.4f}")
    print(f"Pixel-Pooled Specificity:   {pooled_metrics['specificity']:.4f}")
    print(f"Pixel-Pooled Accuracy:      {pooled_metrics['pixel_accuracy']:.4f}")
    print(f"Total Unique Valid Pixels:  {pooled_metrics['valid_pixels']:,}")
    print("-" * 70)
    print("Macro Event Averages (across events):")
    print(f"  Mean Event Flood IoU:     {macro_event_flood_iou:.4f}")
    print(f"  Mean Event Macro mIoU:    {macro_event_mean_iou:.4f}")
    print("-" * 70)
    print(df_per_event[["event", "chip_count", "flood_iou", "bg_iou", "mean_iou", "flood_dice", "flood_precision", "flood_recall"]].to_string(index=False))
    print("=" * 70)
    print(f"Saved metrics JSON to: {json_path}")
    print(f"Saved per-event CSV to: {csv_path}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--experiment", type=str, default="E0")
    parser.add_argument("--max_chips", type=int, default=None)
    args = parser.parse_args()
    
    evaluate_source_chips(split=args.split, experiment_name=args.experiment, max_chips=args.max_chips)
