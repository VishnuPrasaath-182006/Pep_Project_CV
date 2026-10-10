"""Single Fold Training Script for SegFormer-B2 on FloodPlanet."""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from src.data.inspect_dataset import inspect_dataset
from src.data.splits import generate_loeo_splits
from src.training.trainer import FoldTrainer
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("TrainSingleFold")


def train_single_fold(
    config_path: str = "configs/segformer_b2_loeo.yaml",
    fold_idx: int = 0,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate SegFormer-B2 on a single LOEO fold."""
    config = load_config(config_path)
    manifest_path = os.path.join(config["paths"]["output_dir"], "dataset_manifest.csv")
    splits_file = os.path.join(config["paths"]["splits_dir"], "loeo_splits.json")

    # Run inspection if manifest not found
    if not os.path.exists(manifest_path):
        logger.info("Dataset manifest not found. Running inspection first...")
        inspect_dataset(
            dataset_root=config["data"]["dataset_root"],
            output_dir=config["paths"]["output_dir"],
            sensor=config["data"]["sensor"],
        )

    # Generate splits if not found
    if not os.path.exists(splits_file):
        logger.info("LOEO splits not found. Generating splits...")
        generate_loeo_splits(
            dataset_manifest_path=manifest_path,
            splits_output_dir=config["paths"]["splits_dir"],
            val_ratio=config["evaluation"].get("inner_val_ratio", 0.17),
            seed=config["training"].get("seed", 42),
        )

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    if fold_idx < 0 or fold_idx >= len(splits_data["folds"]):
        raise ValueError(f"Fold index {fold_idx} invalid. Total folds available: {len(splits_data['folds'])}")

    fold_spec = splits_data["folds"][fold_idx]
    manifest_df = pd.read_csv(manifest_path)

    trainer = FoldTrainer(
        config=config,
        fold_spec=fold_spec,
        manifest_df=manifest_df,
    )

    results = trainer.train_and_evaluate(force_retrain=force_retrain)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer-B2 on one LOEO fold.")
    parser.add_argument("--config", type=str, default="configs/segformer_b2_loeo.yaml")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0 to 18)")
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain even if results exist")
    args = parser.parse_args()

    train_single_fold(args.config, args.fold, args.force_retrain)
