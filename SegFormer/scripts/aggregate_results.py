#!/usr/bin/env python3
"""Aggregate completed LOEO-CV fold results and update summary metrics."""

import argparse
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.run_loeo import aggregate_loeo_results
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Aggregate LOEO-CV results.")
    parser.add_argument("--config", type=str, default="configs/segformer_b2_loeo.yaml", help="Path to config YAML")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing fold results")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = args.results_dir or config["paths"]["results_dir"]

    summary = aggregate_loeo_results(results_dir=results_dir)
    macro = summary.get("event_macro_mean", {})
    pooled = summary.get("pixel_pooled_metrics", {})

    print("\n" + "=" * 65)
    print("             AGGREGATED LOEO-CV BENCHMARK RESULTS")
    print("=" * 65)
    print(f"  Completed Folds          : {summary.get('completed_folds', 0)} / 19")
    print("-" * 65)
    print("  EVENT-MACRO METRICS (Unweighted mean across events)")
    print(f"  * Macro Accuracy         : {macro.get('accuracy', 0.0):.4f} ({macro.get('accuracy', 0.0)*100:.2f}%)")
    print(f"  * Macro Precision        : {macro.get('precision', 0.0):.4f} ({macro.get('precision', 0.0)*100:.2f}%)")
    print(f"  * Macro Recall           : {macro.get('recall', 0.0):.4f} ({macro.get('recall', 0.0)*100:.2f}%)")
    print(f"  * Macro F1 / Dice        : {macro.get('f1', 0.0):.4f} ({macro.get('f1', 0.0)*100:.2f}%)")
    print(f"  * Macro Specificity      : {macro.get('specificity', 0.0):.4f} ({macro.get('specificity', 0.0)*100:.2f}%)")
    print(f"  * Macro Flood IoU        : {macro.get('flood_iou', 0.0):.4f} ({macro.get('flood_iou', 0.0)*100:.2f}%)")
    print(f"  * Macro mIoU             : {macro.get('miou', 0.0):.4f} ({macro.get('miou', 0.0)*100:.2f}%)")
    print("-" * 65)
    print("  GLOBAL PIXEL-POOLED METRICS (All 383M pixels evaluated)")
    print(f"  * Pooled Accuracy        : {pooled.get('accuracy', 0.0):.4f} ({pooled.get('accuracy', 0.0)*100:.2f}%)")
    print(f"  * Pooled Precision       : {pooled.get('precision', 0.0):.4f} ({pooled.get('precision', 0.0)*100:.2f}%)")
    print(f"  * Pooled Recall          : {pooled.get('recall', 0.0):.4f} ({pooled.get('recall', 0.0)*100:.2f}%)")
    print(f"  * Pooled F1 / Dice       : {pooled.get('f1', 0.0):.4f} ({pooled.get('f1', 0.0)*100:.2f}%)")
    print(f"  * Pooled Flood IoU       : {pooled.get('flood_iou', 0.0):.4f} ({pooled.get('flood_iou', 0.0)*100:.2f}%)")
    print("=" * 65)
    print("Tip: Run `python scripts/evaluate_results.py` to see the per-event breakdown table.\n")


if __name__ == "__main__":
    main()
