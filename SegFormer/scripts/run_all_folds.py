#!/usr/bin/env python3
"""Run all folds of SegFormer-B2 LOEO-CV."""

import argparse
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.run_loeo import run_full_loeo_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run all folds of LOEO cross-validation.")
    parser.add_argument("--config", type=str, default="configs/segformer_b2_loeo.yaml", help="Path to config YAML")
    parser.add_argument("--start_fold", type=int, default=0, help="Starting fold index (0-indexed)")
    parser.add_argument("--end_fold", type=int, default=None, help="Ending fold index (exclusive)")
    parser.add_argument("--resume", action="store_true", help="Resume without retraining completed folds")
    args = parser.parse_args()

    summary = run_full_loeo_pipeline(
        config_path=args.config,
        start_fold=args.start_fold,
        end_fold=args.end_fold,
        force_retrain=not args.resume,
    )
    print("\n=== LOEO Cross-Validation Summary ===")
    print(f"Completed Folds: {summary.get('completed_folds', 'N/A')}")
    print(f"Macro Flood IoU: {summary.get('event_macro_mean', {}).get('flood_iou', 'N/A')}")
    print(f"Macro mIoU: {summary.get('event_macro_mean', {}).get('miou', 'N/A')}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
