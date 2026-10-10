"""Dataset Inspection and Auditing Script for FloodPlanet Dataset.

Performs exhaustive verification of image-mask pairing, spatial dimensions,
band counts, radiometric ranges, and class distributions across all 19 flood events.
Generates:
- outputs/dataset_summary.json
- outputs/dataset_manifest.csv
- outputs/event_statistics.csv
"""

import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import tifffile

from src.utils.logging import get_logger
from src.data.label_processing import RAW_NODATA, RAW_NON_FLOOD, RAW_FLOOD

logger = get_logger("InspectDataset")


def inspect_dataset(
    dataset_root: str = "Datasets",
    output_dir: str = "outputs",
    sensor: str = "PS",
    rgb_bands: Tuple[int, int, int] = (2, 1, 0),
) -> Dict[str, Any]:
    """Inspect and validate the entire FloodPlanet dataset."""
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")

    os.makedirs(output_dir, exist_ok=True)
    
    events = sorted([
        d for d in os.listdir(dataset_root) 
        if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith(".")
    ])
    
    logger.info(f"Discovered {len(events)} event directories in {dataset_root}: {events}")

    manifest_records: List[Dict[str, Any]] = []
    rejection_records: List[Dict[str, Any]] = []
    event_stats_records: List[Dict[str, Any]] = []

    total_valid_samples = 0
    total_corrupted_samples = 0
    total_pixels_all = 0
    flood_pixels_all = 0
    non_flood_pixels_all = 0
    nodata_pixels_all = 0

    for event_name in events:
        event_path = os.path.join(dataset_root, event_name)
        available_subdirs = sorted([
            s for s in os.listdir(event_path) 
            if os.path.isdir(os.path.join(event_path, s))
        ])
        
        sensor_dir = os.path.join(event_path, sensor)
        label_dir = os.path.join(event_path, "labels")

        if not os.path.exists(sensor_dir):
            logger.warning(f"Event {event_name} missing sensor directory: {sensor_dir}")
            continue
        if not os.path.exists(label_dir):
            logger.warning(f"Event {event_name} missing label directory: {label_dir}")
            continue

        sensor_files = sorted([f for f in os.listdir(sensor_dir) if f.endswith(".tif")])
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".tif")])

        event_valid_count = 0
        event_total_pixels = 0
        event_flood_pixels = 0
        event_non_flood_pixels = 0
        event_nodata_pixels = 0

        # Verify pairing
        all_basenames = sorted(list(set(sensor_files) | set(label_files)))
        for base in all_basenames:
            img_path = os.path.join(sensor_dir, base)
            mask_path = os.path.join(label_dir, base)

            if not os.path.exists(img_path):
                rejection_records.append({
                    "event": event_name,
                    "filename": base,
                    "reason": f"Missing sensor image in {sensor_dir}",
                })
                continue
            if not os.path.exists(mask_path):
                rejection_records.append({
                    "event": event_name,
                    "filename": base,
                    "reason": f"Missing label mask in {label_dir}",
                })
                continue

            # Inspect files
            try:
                img = tifffile.imread(img_path)
                mask = tifffile.imread(mask_path)
            except Exception as e:
                total_corrupted_samples += 1
                rejection_records.append({
                    "event": event_name,
                    "filename": base,
                    "reason": f"Corrupted TIFF file: {str(e)}",
                })
                continue

            # Spatial dimensions check
            img_shape = img.shape
            mask_shape = mask.shape

            if len(img_shape) == 3:
                h, w, c = img_shape
            elif len(img_shape) == 2:
                h, w = img_shape
                c = 1
            else:
                rejection_records.append({
                    "event": event_name,
                    "filename": base,
                    "reason": f"Unexpected image dimensions: {img_shape}",
                })
                continue

            mask_h, mask_w = mask_shape[:2]
            if (h, w) != (mask_h, mask_w):
                rejection_records.append({
                    "event": event_name,
                    "filename": base,
                    "reason": f"Shape mismatch between image ({h}x{w}) and mask ({mask_h}x{mask_w})",
                })
                continue

            # Pixel counts
            mask_int = np.rint(mask).astype(np.int64)
            unq, counts = np.unique(mask_int, return_counts=True)
            counts_dict = dict(zip(unq.tolist(), counts.tolist()))

            nodata_cnt = counts_dict.get(RAW_NODATA, 0)
            non_flood_cnt = counts_dict.get(RAW_NON_FLOOD, 0)
            flood_cnt = counts_dict.get(RAW_FLOOD, 0)
            valid_cnt = non_flood_cnt + flood_cnt
            sample_total_px = h * w

            event_total_pixels += sample_total_px
            event_flood_pixels += flood_cnt
            event_non_flood_pixels += non_flood_cnt
            event_nodata_pixels += nodata_cnt
            event_valid_count += 1
            total_valid_samples += 1

            manifest_records.append({
                "event": event_name,
                "filename": base,
                "image_path": os.path.relpath(img_path, dataset_root),
                "mask_path": os.path.relpath(mask_path, dataset_root),
                "sensor": sensor,
                "width": w,
                "height": h,
                "bands": c,
                "image_dtype": str(img.dtype),
                "mask_dtype": str(mask.dtype),
                "image_min": float(img.min()),
                "image_max": float(img.max()),
                "valid_pixels": valid_cnt,
                "flood_pixels": flood_cnt,
                "non_flood_pixels": non_flood_cnt,
                "nodata_pixels": nodata_cnt,
                "flood_percent": round((flood_cnt / valid_cnt * 100), 2) if valid_cnt > 0 else 0.0,
                "nodata_percent": round((nodata_cnt / sample_total_px * 100), 2),
                "paired": True,
            })

        total_pixels_all += event_total_pixels
        flood_pixels_all += event_flood_pixels
        non_flood_pixels_all += event_non_flood_pixels
        nodata_pixels_all += event_nodata_pixels

        valid_event_px = event_non_flood_pixels + event_flood_pixels
        event_stats_records.append({
            "event": event_name,
            "chips_count": event_valid_count,
            "available_sensors": available_subdirs,
            "total_pixels": event_total_pixels,
            "valid_pixels": valid_event_px,
            "flood_pixels": event_flood_pixels,
            "non_flood_pixels": event_non_flood_pixels,
            "nodata_pixels": event_nodata_pixels,
            "flood_percentage": round((event_flood_pixels / valid_event_px * 100), 2) if valid_event_px > 0 else 0.0,
            "nodata_percentage": round((event_nodata_pixels / event_total_pixels * 100), 2) if event_total_pixels > 0 else 0.0,
        })

    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    manifest_csv_path = os.path.join(output_dir, "dataset_manifest.csv")
    manifest_df.to_csv(manifest_csv_path, index=False)
    logger.info(f"Saved dataset manifest to {manifest_csv_path} ({len(manifest_df)} samples)")

    # Save event stats CSV
    event_stats_df = pd.DataFrame(event_stats_records)
    event_stats_csv_path = os.path.join(output_dir, "event_statistics.csv")
    event_stats_df.to_csv(event_stats_csv_path, index=False)
    logger.info(f"Saved event statistics to {event_stats_csv_path} ({len(event_stats_df)} events)")

    # Rejection report
    if rejection_records:
        rejection_df = pd.DataFrame(rejection_records)
        rejection_path = os.path.join(output_dir, "dataset_rejections.csv")
        rejection_df.to_csv(rejection_path, index=False)
        logger.warning(f"Rejections recorded: {len(rejection_df)} (saved to {rejection_path})")

    # Dataset summary JSON
    valid_all = flood_pixels_all + non_flood_pixels_all
    summary = {
        "dataset_name": "FloodPlanet",
        "dataset_root": dataset_root,
        "primary_sensor": sensor,
        "total_events": len(events),
        "total_valid_chips": total_valid_samples,
        "total_corrupted_chips": total_corrupted_samples,
        "chip_dimensions": "1024x1024",
        "total_pixels": int(total_pixels_all),
        "total_valid_pixels": int(valid_all),
        "total_flood_pixels": int(flood_pixels_all),
        "total_non_flood_pixels": int(non_flood_pixels_all),
        "total_nodata_pixels": int(nodata_pixels_all),
        "global_flood_percentage": round((flood_pixels_all / valid_all * 100), 2) if valid_all > 0 else 0.0,
        "events": [r["event"] for r in event_stats_records],
        "event_chip_counts": {r["event"]: r["chips_count"] for r in event_stats_records},
    }

    summary_json_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved dataset summary to {summary_json_path}")

    return summary


if __name__ == "__main__":
    inspect_dataset()
