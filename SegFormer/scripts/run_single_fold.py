#!/usr/bin/env python3
"""Run a single fold of SegFormer-B2 LOEO-CV training."""

import argparse
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train import train_single_fold


def main():
    parser = argparse.ArgumentParser(description="Run single fold training for SegFormer-B2.")
    parser.add_argument("--config", type=str, default="configs/segformer_b2_loeo.yaml", help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0 to 18)")
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain if results already exist")
    args = parser.parse_args()

    results = train_single_fold(
        config_path=args.config,
        fold_idx=args.fold,
        force_retrain=args.force_retrain,
    )
    print("\n=== Fold Training Complete ===")
    print(f"Fold: {results['fold_id']} | Test Event: {results['test_event']}")
    print(f"Best Val Epoch: {results['best_validation_epoch']} (Val Flood IoU: {results['best_val_flood_iou']})")
    print(f"Test Flood IoU: {results['test_metrics']['flood_iou']}")
    print(f"Test mIoU: {results['test_metrics']['miou']}")
    print("==============================\n")


if __name__ == "__main__":
    main()
