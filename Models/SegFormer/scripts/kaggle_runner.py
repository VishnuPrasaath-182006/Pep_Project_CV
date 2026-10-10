#!/usr/bin/env python3
"""Kaggle GPU Runner Script for SegFormer-B2 FloodPlanet LOEO-CV.

Features:
- Auto-detects Kaggle dataset path (/kaggle/input/floodplanet/...)
- Auto-configures CUDA AMP and 4 vCPU DataLoader workers
- Safe resumption if Kaggle 9-hour session limit is reached
- Persists all checkpoints and figures to /kaggle/working/outputs/
- Produces event_metrics.csv, summary_metrics.json, and comparative bar charts
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.run_loeo import run_full_loeo_pipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("KaggleRunner")


def run_on_kaggle():
    logger.info("Initializing Kaggle GPU LOEO-CV Pipeline...")
    
    config_file = "configs/kaggle_gpu.yaml"
    if not os.path.exists(config_file):
        config_file = "configs/segformer_b2_loeo.yaml"
        
    config = load_config(config_file)
    logger.info(f"Loaded config: {config_file}")
    logger.info(f"Dataset root: {config['data']['dataset_root']}")
    logger.info(f"Output directory: {config['paths']['output_dir']}")

    # Run full 19 folds with safe resumption
    summary = run_full_loeo_pipeline(
        config_path=config_file,
        start_fold=0,
        end_fold=19,
        force_retrain=False,  # Resume any completed folds
    )

    logger.info("=== Kaggle LOEO Execution Finished ===")
    logger.info(f"Completed Folds: {summary.get('completed_folds')}")
    logger.info(f"Macro Flood IoU: {summary.get('event_macro_mean', {}).get('flood_iou')}")
    logger.info(f"Macro mIoU: {summary.get('event_macro_mean', {}).get('miou')}")


if __name__ == "__main__":
    run_on_kaggle()
