"""Master LOEO Cross-Validation Execution Engine.

Iterates through all 19 folds, trains each for exactly 10 epochs,
evaluates on held-out test events, and aggregates macro and pixel-pooled metrics.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.inspect_dataset import inspect_dataset
from src.data.splits import generate_loeo_splits
from src.training.trainer import FoldTrainer
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.visualization.visualize import plot_event_metrics_bar

logger = get_logger("RunLOEO")


def run_full_loeo_pipeline(
    config_path: str = "configs/segformer_b2_loeo.yaml",
    start_fold: int = 0,
    end_fold: Optional[int] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Execute complete Leave-One-Event-Out cross-validation pipeline."""
    config = load_config(config_path)
    output_dir = config["paths"]["output_dir"]
    results_dir = config["paths"]["results_dir"]
    splits_dir = config["paths"]["splits_dir"]

    manifest_path = os.path.join(output_dir, "dataset_manifest.csv")
    splits_file = os.path.join(splits_dir, "loeo_splits.json")

    # Ensure dataset manifest
    if not os.path.exists(manifest_path):
        inspect_dataset(
            dataset_root=config["data"]["dataset_root"],
            output_dir=output_dir,
            sensor=config["data"]["sensor"],
        )

    # Ensure splits
    if not os.path.exists(splits_file):
        generate_loeo_splits(
            dataset_manifest_path=manifest_path,
            splits_output_dir=splits_dir,
            val_ratio=config["evaluation"].get("inner_val_ratio", 0.17),
            seed=config["training"].get("seed", 42),
        )

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    all_folds = splits_data["folds"]
    total_folds = len(all_folds)
    manifest_df = pd.read_csv(manifest_path)

    if end_fold is None:
        end_fold = total_folds

    logger.info(f"Starting LOEO Cross-Validation: running folds {start_fold} to {end_fold - 1} of {total_folds}")

    fold_results_list = []
    global_cm = np.zeros((2, 2), dtype=np.int64)

    for f_idx in range(start_fold, min(end_fold, total_folds)):
        fold_spec = all_folds[f_idx]
        logger.info(f"========== Processing {fold_spec['fold_id']} (Test Event: {fold_spec['test_event']}) ==========")

        trainer = FoldTrainer(
            config=config,
            fold_spec=fold_spec,
            manifest_df=manifest_df,
        )

        res = trainer.train_and_evaluate(force_retrain=force_retrain)
        fold_results_list.append(res)

        test_m = res["test_metrics"]
        cm = np.array(test_m["confusion_matrix"], dtype=np.int64)
        global_cm += cm

    # Aggregate and generate summary tables
    summary = aggregate_loeo_results(results_dir=results_dir, all_folds=all_folds)
    return summary


def aggregate_loeo_results(results_dir: str = "outputs/results", all_folds: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Aggregate all completed fold results into event_metrics.csv and summary_metrics.json."""
    os.makedirs(results_dir, exist_ok=True)
    
    event_rows = []
    global_cm = np.zeros((2, 2), dtype=np.int64)
    completed_count = 0

    import re
    if all_folds is None:
        # Load all available primary fold json results (exclude fold_XX_test_metrics.json)
        fold_files = sorted([f for f in os.listdir(results_dir) if re.match(r"^fold_\d{2}_metrics\.json$", f)])
    else:
        fold_files = [f"{f['fold_id']}_metrics.json" for f in all_folds]

    for fname in fold_files:
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            res = json.load(f)

        m = res.get("test_metrics") or res.get("metrics")
        if not m:
            continue
        cm = np.array(m["confusion_matrix"], dtype=np.int64)
        global_cm += cm
        completed_count += 1

        event_rows.append({
            "fold_id": res["fold_id"],
            "test_event": res["test_event"],
            "train_events_count": res["training_event_count"],
            "val_events_count": res["validation_event_count"],
            "test_chips_count": res["test_image_count"],
            "flood_iou": m["flood_iou"],
            "non_flood_iou": m["non_flood_iou"],
            "miou": m["miou"],
            "f1_score": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "accuracy": m["accuracy"],
            "tp": m["tp"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "best_val_epoch": res["best_validation_epoch"],
            "best_val_flood_iou": res.get("best_val_flood_iou", 0.0),
            "status": "Completed",
        })

    event_df = pd.DataFrame(event_rows)
    event_csv_path = os.path.join(results_dir, "event_metrics.csv")
    event_df.to_csv(event_csv_path, index=False)
    logger.info(f"Saved event metrics to {event_csv_path} ({len(event_df)} completed events)")

    if len(event_df) == 0:
        return {"status": "No completed folds yet"}

    # Compute Macro statistics across completed events
    macro_mean = {
        "flood_iou": round(float(event_df["flood_iou"].mean()), 4),
        "non_flood_iou": round(float(event_df["non_flood_iou"].mean()), 4),
        "miou": round(float(event_df["miou"].mean()), 4),
        "f1": round(float(event_df["f1_score"].mean()), 4),
        "precision": round(float(event_df["precision"].mean()), 4),
        "recall": round(float(event_df["recall"].mean()), 4),
        "specificity": round(float(event_df["specificity"].mean()), 4),
        "accuracy": round(float(event_df["accuracy"].mean()), 4),
    }

    macro_std = {
        "flood_iou": round(float(event_df["flood_iou"].std()), 4) if len(event_df) > 1 else 0.0,
        "miou": round(float(event_df["miou"].std()), 4) if len(event_df) > 1 else 0.0,
        "f1": round(float(event_df["f1_score"].std()), 4) if len(event_df) > 1 else 0.0,
    }

    # Compute Pixel-Pooled Global Metrics
    from src.evaluation.metrics import ConfusionMatrixAccumulator
    acc = ConfusionMatrixAccumulator()
    acc.matrix = global_cm
    pixel_pooled = acc.compute_metrics()

    summary_data = {
        "completed_folds": completed_count,
        "total_folds": len(all_folds) if all_folds else completed_count,
        "event_macro_mean": macro_mean,
        "event_macro_std": macro_std,
        "pixel_pooled_metrics": pixel_pooled,
    }

    summary_json_path = os.path.join(results_dir, "summary_metrics.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Plot event bar charts
    vis_dir = os.path.join(results_dir, "..", "visualizations")
    plot_event_metrics_bar(
        events=event_df["test_event"].tolist(),
        metric_values=event_df["flood_iou"].tolist(),
        metric_name="Flood IoU",
        save_path=os.path.join(vis_dir, "event_flood_iou_comparison.png"),
    )
    plot_event_metrics_bar(
        events=event_df["test_event"].tolist(),
        metric_values=event_df["miou"].tolist(),
        metric_name="mIoU",
        save_path=os.path.join(vis_dir, "event_miou_comparison.png"),
    )

    logger.info(f"Saved summary metrics to {summary_json_path}")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full LOEO cross-validation pipeline.")
    parser.add_argument("--config", type=str, default="configs/segformer_b2_loeo.yaml")
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--end_fold", type=int, default=None)
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()

    run_full_loeo_pipeline(
        config_path=args.config,
        start_fold=args.start_fold,
        end_fold=args.end_fold,
        force_retrain=args.force_retrain,
    )
