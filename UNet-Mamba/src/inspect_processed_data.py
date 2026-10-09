import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from config import load_config, BASE_DIR
from utils import get_logger

def inspect_processed_data(config=None, output_dir=None):
    if config is None:
        config = load_config()
    logger = get_logger("InspectProcessedData")
    
    processed_dir = BASE_DIR / config['data']['processed_dir']
    if output_dir is None:
        output_dir = BASE_DIR / "results"
    else:
        output_dir = Path(output_dir)
        
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures" / "processed_comparisons"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    summary = {}
    total_patches = 0
    
    for split_name in ["train", "val", "test"]:
        patch_csv = processed_dir / f"{split_name}_patches.csv"
        if not patch_csv.exists():
            logger.warning(f"Patch manifest not found: {patch_csv}")
            continue
            
        df_patches = pd.read_csv(patch_csv)
        num_patches = len(df_patches)
        total_patches += num_patches
        
        # Spot check first 10 patches for shape, dtype, nan/inf
        has_nan = False
        has_inf = False
        valid_shapes = True
        unique_mask_values = set()
        
        for idx in range(min(20, num_patches)):
            patch_path = df_patches.iloc[idx]["patch_path"]
            data = np.load(patch_path)
            img = data["image"]  # (4, 512, 512)
            mask = data["mask"]  # (512, 512)
            
            if img.shape != (4, 512, 512) or mask.shape != (512, 512):
                valid_shapes = False
            if np.isnan(img).any() or np.isnan(mask).any():
                has_nan = True
            if np.isinf(img).any() or np.isinf(mask).any():
                has_inf = True
                
            unique_mask_values.update(np.unique(mask).tolist())
            
        summary[split_name] = {
            "patch_count": num_patches,
            "valid_shapes": valid_shapes,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "mask_unique_values": sorted(list(unique_mask_values))
        }
        
    # Generate visualization for a few sample patches
    train_csv = processed_dir / "train_patches.csv"
    if train_csv.exists():
        df_train = pd.read_csv(train_csv)
        sample_indices = [0, min(10, len(df_train)-1), min(50, len(df_train)-1)]
        for s_idx in sample_indices:
            row = df_train.iloc[s_idx]
            data = np.load(row["patch_path"])
            img = data["image"]
            mask = data["mask"]
            
            rgb = img[:3, :, :].transpose(1, 2, 0).astype(np.float32)
            rgb_norm = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(rgb_norm)
            axes[0].set_title(f"Processed Patch Image 512x512 ({row['event']})")
            axes[0].axis("off")
            
            mask_viz = np.zeros_like(mask, dtype=np.int32)
            mask_viz[mask == 0] = 0
            mask_viz[mask == 1] = 1
            mask_viz[mask == -1] = -1
            
            axes[1].imshow(mask_viz, cmap="coolwarm", vmin=-1, vmax=1)
            axes[1].set_title(f"Patch Mask (Blue=BG, Red=Flood, Grey=Ignore)")
            axes[1].axis("off")
            
            plt.tight_layout()
            fig_path = figures_dir / f"processed_sample_{row['event']}_{row['stem']}_p{row['patch_idx']}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
    summary_json = reports_dir / "processed_data_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    md_content = f"""# Processed Data Inspection Report

## Overview
- **Total Processed Patches (512x512)**: {total_patches}
- **Patch Size**: 512 × 512
- **Stride**: 256
- **Input Channels**: 4 (uint16)
- **Mask Encoding**: 0 = Background, 1 = Flood, -1 = Ignored/Nodata

## Split Patch Breakdown
"""
    for split_name, s_data in summary.items():
        md_content += f"- **{split_name.capitalize()}**: {s_data['patch_count']} patches | Valid Shapes: {s_data['valid_shapes']} | Unique Mask Values: {s_data['mask_unique_values']}\n"
        
    md_report = reports_dir / "processed_data_inspection.md"
    with open(md_report, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logger.info(f"Processed data inspection completed! Reports saved to {reports_dir}")

if __name__ == "__main__":
    inspect_processed_data()
