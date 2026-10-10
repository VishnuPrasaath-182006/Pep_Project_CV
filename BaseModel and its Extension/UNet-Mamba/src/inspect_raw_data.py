import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import rasterio
except ImportError:
    import PIL.Image as rasterio  # Fallback if needed

from config import load_config, ALL_19_EVENTS, BASE_DIR
from utils import get_logger

def inspect_raw_dataset(config=None, output_dir=None):
    if config is None:
        config = load_config()
    logger = get_logger("InspectRawData")
    
    raw_dataset_dir = Path(config['data']['raw_dataset_dir'])
    if not raw_dataset_dir.is_absolute():
        raw_dataset_dir = BASE_DIR / raw_dataset_dir
        
    sensor = config['data']['sensor']  # "PS"
    logger.info(f"Inspecting raw dataset at: {raw_dataset_dir} for sensor {sensor}")
    
    if output_dir is None:
        output_dir = BASE_DIR / "results"
    else:
        output_dir = Path(output_dir)
        
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures" / "raw_samples"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    inventory_records = []
    event_counts = {evt: {"images": 0, "masks": 0, "matched": 0} for evt in ALL_19_EVENTS}
    
    total_bg_pixels = 0
    total_flood_pixels = 0
    total_nodata_pixels = 0
    
    # Iterate through each event folder
    for event_name in ALL_19_EVENTS:
        event_dir = raw_dataset_dir / event_name
        if not event_dir.exists():
            logger.warning(f"Event directory not found: {event_dir}")
            continue
            
        ps_dir = event_dir / sensor
        mask_dir = event_dir / "labels"
        if not mask_dir.exists():
            mask_dir = event_dir / "Masks"
        
        ps_files = sorted(glob.glob(str(ps_dir / "*.tif"))) + sorted(glob.glob(str(ps_dir / "*.png")))
        mask_files = sorted(glob.glob(str(mask_dir / "*.tif"))) + sorted(glob.glob(str(mask_dir / "*.png")))
        
        event_counts[event_name]["images"] = len(ps_files)
        event_counts[event_name]["masks"] = len(mask_files)
        
        # Match image and mask files by base filename
        ps_map = {Path(p).stem: p for p in ps_files}
        mask_map = {Path(m).stem: m for m in mask_files}
        
        matched_stems = sorted(list(set(ps_map.keys()).intersection(set(mask_map.keys()))))
        event_counts[event_name]["matched"] = len(matched_stems)
        
        for stem in matched_stems:
            img_path = ps_map[stem]
            mask_path = mask_map[stem]
            
            with rasterio.open(img_path) as src_img:
                height, width = src_img.height, src_img.width
                channels = src_img.count
                dtype = str(src_img.dtypes[0])
                # Read 4 bands
                img_data = src_img.read()
                img_min = float(img_data.min())
                img_max = float(img_data.max())
                img_mean = float(img_data.mean())
                img_std = float(img_data.std())
                
            with rasterio.open(mask_path) as src_mask:
                mask_data = src_mask.read(1)
                unique_vals = np.unique(mask_data).tolist()
                
                # Class 1 = Background, Class 2 = Flood, 0/-1 = Nodata
                bg_cnt = int(np.sum(mask_data == 1))
                flood_cnt = int(np.sum(mask_data == 2))
                nodata_cnt = int(np.sum((mask_data != 1) & (mask_data != 2)))
                
                total_bg_pixels += bg_cnt
                total_flood_pixels += flood_cnt
                total_nodata_pixels += nodata_cnt
                
            inventory_records.append({
                "event": event_name,
                "stem": stem,
                "img_path": img_path,
                "mask_path": mask_path,
                "height": height,
                "width": width,
                "channels": channels,
                "dtype": dtype,
                "img_min": img_min,
                "img_max": img_max,
                "img_mean": img_mean,
                "img_std": img_std,
                "mask_unique_values": str(unique_vals),
                "bg_pixels": bg_cnt,
                "flood_pixels": flood_cnt,
                "nodata_pixels": nodata_cnt
            })
            
    df_inventory = pd.DataFrame(inventory_records)
    inventory_csv = reports_dir / "raw_data_inventory.csv"
    df_inventory.to_csv(inventory_csv, index=False)
    logger.info(f"Saved raw data inventory to {inventory_csv}")
    
    # Event counts CSV
    df_events = pd.DataFrame([
        {"event": k, "images": v["images"], "masks": v["masks"], "matched": v["matched"]}
        for k, v in event_counts.items()
    ])
    event_counts_csv = reports_dir / "event_counts.csv"
    df_events.to_csv(event_counts_csv, index=False)
    
    # Class distribution CSV
    total_valid_pixels = total_bg_pixels + total_flood_pixels
    df_class = pd.DataFrame([
        {
            "class_name": "Background (Non-flooded)",
            "raw_value": 1,
            "mapped_value": 0,
            "pixel_count": total_bg_pixels,
            "percentage": round(total_bg_pixels / max(1, total_valid_pixels) * 100, 2)
        },
        {
            "class_name": "Flood (Flooded)",
            "raw_value": 2,
            "mapped_value": 1,
            "pixel_count": total_flood_pixels,
            "percentage": round(total_flood_pixels / max(1, total_valid_pixels) * 100, 2)
        },
        {
            "class_name": "Nodata / Invalid",
            "raw_value": "Other",
            "mapped_value": -1,
            "pixel_count": total_nodata_pixels,
            "percentage": 0.0
        }
    ])
    class_csv = reports_dir / "class_distribution.csv"
    df_class.to_csv(class_csv, index=False)
    
    # Generate JSON summary
    summary = {
        "sensor": sensor,
        "total_matched_scenes": len(df_inventory),
        "total_events": len(ALL_19_EVENTS),
        "event_summary": event_counts,
        "pixel_distribution": {
            "background_pixels": total_bg_pixels,
            "flood_pixels": total_flood_pixels,
            "nodata_pixels": total_nodata_pixels,
            "background_percent": round(total_bg_pixels / max(1, total_valid_pixels) * 100, 2),
            "flood_percent": round(total_flood_pixels / max(1, total_valid_pixels) * 100, 2)
        },
        "imagery_specs": {
            "channels": int(df_inventory["channels"].iloc[0]) if len(df_inventory) > 0 else 4,
            "height": int(df_inventory["height"].iloc[0]) if len(df_inventory) > 0 else 1024,
            "width": int(df_inventory["width"].iloc[0]) if len(df_inventory) > 0 else 1024,
            "dtype": str(df_inventory["dtype"].iloc[0]) if len(df_inventory) > 0 else "uint16"
        }
    }
    summary_json = reports_dir / "raw_data_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    # Generate Markdown report
    md_content = f"""# Raw Data Inspection Report — FloodPlanet ({sensor} Sensor)

## Overview
- **Sensor/Product**: PlanetScope ({sensor}) 4-band imagery
- **Total Events**: {len(ALL_19_EVENTS)}
- **Total Matched Scenes (1024x1024)**: {len(df_inventory)}
- **Input Channels**: {summary['imagery_specs']['channels']} (Bands: 1-Red, 2-Green, 3-Blue, 4-NIR)
- **Native Data Type**: {summary['imagery_specs']['dtype']}

## Class Distribution
- **Background (Class 0, Raw 1)**: {total_bg_pixels:,} pixels ({summary['pixel_distribution']['background_percent']}%)
- **Flood (Class 1, Raw 2)**: {total_flood_pixels:,} pixels ({summary['pixel_distribution']['flood_percent']}%)
- **Nodata (Ignored -1)**: {total_nodata_pixels:,} pixels

## Event Scene Counts
| Event | Matched Scenes |
|---|---|
"""
    for _, row in df_events.iterrows():
        md_content += f"| {row['event']} | {row['matched']} |\n"
        
    md_report = reports_dir / "raw_data_inspection.md"
    with open(md_report, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    # Generate Raw Sample Visualizations for representative scenes
    sample_indices = np.linspace(0, max(0, len(df_inventory) - 1), min(5, len(df_inventory)), dtype=int)
    for idx in sample_indices:
        rec = df_inventory.iloc[idx]
        with rasterio.open(rec["img_path"]) as s_img:
            img = s_img.read()  # (4, H, W)
        with rasterio.open(rec["mask_path"]) as s_mask:
            mask = s_mask.read(1)
            
        # RGB False Color composite (RGB bands 1,2,3 or NIR-R-G 4,1,2)
        rgb = img[:3, :, :].transpose(1, 2, 0).astype(np.float32)
        # Simple stretch for visualization
        rgb_norm = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_norm)
        axes[0].set_title(f"Raw PlanetScope RGB ({rec['event']})")
        axes[0].axis("off")
        
        cmap = plt.cm.get_cmap("bwr", 3)
        # Remap raw 1->0 (bg), 2->1 (flood)
        mask_viz = np.zeros_like(mask, dtype=np.int32)
        mask_viz[mask == 1] = 0
        mask_viz[mask == 2] = 1
        
        axes[1].imshow(mask_viz, cmap="coolwarm", vmin=0, vmax=1)
        axes[1].set_title(f"Ground Truth Mask (Blue=BG, Red=Flood)")
        axes[1].axis("off")
        
        plt.tight_layout()
        fig_path = figures_dir / f"raw_sample_{rec['event']}_{rec['stem']}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
    logger.info(f"Completed Phase 2 inspection. Reports saved to {reports_dir}")

if __name__ == "__main__":
    inspect_raw_dataset()
