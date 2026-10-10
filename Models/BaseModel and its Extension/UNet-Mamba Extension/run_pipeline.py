import os
import sys
import argparse
from pathlib import Path

# Add src/ directory to python path
src_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_dir))

from config import load_config
from utils import get_logger
from inspect_raw_data import inspect_raw_dataset
from split_manager import generate_split_manifests
from preprocessing import preprocess_dataset
from inspect_processed_data import inspect_processed_data
from verify_pipeline import verify_pipeline
from train import run_training
from evaluate import run_evaluation
from visualize import generate_visualizations

def main():
    parser = argparse.ArgumentParser(description="UNet-Mamba Flood Inundation Pipeline Master Runner")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "inspect-raw",
            "split",
            "preprocess",
            "inspect-processed",
            "verify",
            "train",
            "evaluate",
            "visualize",
            "all"
        ],
        help="Pipeline stage to execute"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to custom configuration YAML")
    args = parser.parse_args()
    
    config = load_config(args.config)
    logger = get_logger("MasterRunner")
    stage = args.stage.lower()
    
    logger.info(f"=== UNet-Mamba Pipeline Execution Stage: [{stage.upper()}] ===")
    
    if stage == "inspect-raw":
        inspect_raw_dataset(config)
    elif stage == "split":
        generate_split_manifests(config)
    elif stage == "preprocess":
        preprocess_dataset(config)
    elif stage == "inspect-processed":
        inspect_processed_data(config)
    elif stage == "verify":
        verify_pipeline()
    elif stage == "train":
        run_training(config)
    elif stage == "evaluate":
        run_evaluation(config)
    elif stage == "visualize":
        generate_visualizations(config)
    elif stage == "all":
        logger.info("Running FULL end-to-end UNet-Mamba pipeline...")
        logger.info("Stage 1/8: Inspect Raw Dataset")
        inspect_raw_dataset(config)
        logger.info("Stage 2/8: Split Dataset")
        generate_split_manifests(config)
        logger.info("Stage 3/8: Preprocess Dataset")
        preprocess_dataset(config)
        logger.info("Stage 4/8: Inspect Processed Dataset")
        inspect_processed_data(config)
        logger.info("Stage 5/8: Verify Automated Pipeline")
        verify_pipeline()
        logger.info("Stage 6/8: Train UNet-Mamba Model (21 Epochs)")
        run_training(config)
        logger.info("Stage 7/8: Evaluate Held-Out Test Events")
        run_evaluation(config)
        logger.info("Stage 8/8: Generate Segmentation Visualizations")
        generate_visualizations(config)
        logger.info("=== FULL UNET-MAMBA PIPELINE COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()
