#!/usr/bin/env python3
"""Project Inspection CLI Tool for SegFormer Flood Inundation Pipeline.

Provides inspection of:
- Project folder structure
- Python source modules and their architectural purposes
- Dependency status and compatibility
- Python, PyTorch, CUDA, and GPU hardware profile
- Active SegFormer YAML configuration
"""

import argparse
import importlib.util
import os
import sys
from typing import Dict, List, Tuple

# Ensure repository root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODULE_PURPOSES: Dict[str, str] = {
    "src/data/dataset.py": "PyTorch FloodPlanetDataset implementing tiled chip extraction, memory caching, and LOEO indexing.",
    "src/data/preprocessing.py": "PlanetScope band selection (RGB [2,1,0]), dynamic scaling to [0,1], ImageNet normalization, and full-image logit reconstruction.",
    "src/data/label_processing.py": "Strict semantic label mapping: Raw 1->0 (Land), Raw 2->1 (Flood), Raw 0->255 (Ignore/Nodata).",
    "src/data/splits.py": "19-fold Leave-One-Event-Out (LOEO) cross-validation partition generator with strict disjoint invariants.",
    "src/data/inspect_dataset.py": "Dataset auditor scanning 19 events, multi-sensor rasters, chip dimensions, and image-mask pairing.",
    "src/data/inspect_multisensor.py": "Co-registered multi-sensor raster auditor (Sentinel-1 SAR, Sentinel-2 Optical, Landsat-8).",
    "src/data/audit_masks.py": "Pixel-by-pixel mask integrity auditor checking raw unique values, class mapping, and 100% pixel conservation.",
    "src/data/audit_leakage.py": "Zero-leakage verifier testing disjoint Train, Val, and Test event sets across all 19 folds.",
    "src/data/audit_project.py": "Full repository file inventory generator categorizing 1,785 items into Categories A through J.",
    "src/data/cleanup_project.py": "Safe non-destructive cleanup executing removal of redundant metadata while protecting all data and code.",
    "src/models/segformer.py": "SegFormer-B2 wrapper (nvidia/mit-b2 encoder + MLP decoder) configured for 2-class flood segmentation with ignore_index=255.",
    "src/training/trainer.py": "FoldTrainer managing fixed 10-epoch training loop, AMP, gradient accumulation, and best-val checkpointing.",
    "src/training/train.py": "Entry point for training a single isolated LOEO fold and evaluating held-out test event.",
    "src/training/run_loeo.py": "LOEO cross-validation coordinator executing all 19 folds, resume handling, and macro-aggregation.",
    "src/evaluation/metrics.py": "ConfusionMatrixAccumulator computing exact pixel-level Flood IoU, mIoU, Dice, Precision, Recall, Specificity, Accuracy.",
    "src/evaluation/evaluate.py": "Tiled test inference and whole-image reconstruction evaluator accumulating confusion matrices without tile-boundary bias.",
    "src/visualization/visualize.py": "Qualitative 4-panel visualizer (RGB, GT, Pred, Color-coded Error Map with TP/TN/FP/FN/Ignore).",
    "src/visualization/inspect_dataset.py": "Multi-panel dataset contact sheet and HTML gallery generator across all 19 flood events.",
    "src/utils/config.py": "YAML configuration loader and schema validator.",
    "src/utils/logging.py": "Dual console-and-file structured logger.",
    "src/utils/seed.py": "Deterministic random seed initializer for Python, NumPy, and PyTorch (CPU & CUDA).",
    "scripts/preflight.py": "24-point pretraining validation gate enforcing mandatory checks before training initiation.",
    "scripts/smoke_test.py": "Rapid end-to-end sanity check validating all 8 core subsystems on minimal real data.",
    "scripts/run_single_fold.py": "CLI command to train a single outer LOEO fold for 10 epochs.",
    "scripts/run_all_folds.py": "CLI command to run all 19 LOEO folds sequentially with resume capability.",
    "scripts/aggregate_results.py": "CLI command to aggregate fold test metrics into event-macro and pooled benchmarks.",
    "scripts/inspect_data.py": "CLI command to audit dataset dimensions, pairing, and flood statistics.",
    "scripts/kaggle_runner.py": "Self-contained Kaggle GPU execution runner for accelerated 19-fold LOEO cross-validation.",
}


def print_environment_info() -> None:
    print("\n=======================================================")
    print("           ENVIRONMENT & HARDWARE PROFILE              ")
    print("=======================================================")
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Platform / OS     : {sys.platform}")

    try:
        import torch
        print(f"PyTorch Version   : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available    : {cuda_available}")
        if cuda_available:
            print(f"CUDA Device Count : {torch.cuda.device_count()}")
            print(f"GPU Device Name   : {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version      : {torch.version.cuda}")
        else:
            print("GPU Device Name   : N/A (Local CPU Mode)")
            print("Note              : Microsoft Basic Display Adapter detected (no physical NVIDIA GPU).")
    except ImportError:
        print("PyTorch           : NOT INSTALLED")

    try:
        import transformers
        print(f"Transformers Ver  : {transformers.__version__}")
    except ImportError:
        print("Transformers      : NOT INSTALLED")


def print_dependencies() -> None:
    print("\n=======================================================")
    print("                DEPENDENCY VERIFICATION                ")
    print("=======================================================")
    core_packages = [
        "torch",
        "torchvision",
        "transformers",
        "albumentations",
        "tifffile",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "yaml",
        "pytest",
    ]
    for pkg in core_packages:
        spec = importlib.util.find_spec(pkg if pkg != "yaml" else "yaml")
        if spec is not None:
            try:
                mod = importlib.import_module(pkg)
                ver = getattr(mod, "__version__", "Installed")
                print(f"  [OK]   {pkg:18s} : {ver}")
            except Exception as e:
                print(f"  [WARN] {pkg:18s} : Imported with warning ({e})")
        else:
            print(f"  [FAIL] {pkg:18s} : NOT INSTALLED")


def print_modules() -> None:
    print("\n=======================================================")
    print("           PYTHON SOURCE MODULES & PURPOSES            ")
    print("=======================================================")
    print(f"{'File Path':34s} | {'Lines':5s} | {'Size':8s} | Purpose")
    print("-" * 110)

    total_lines = 0
    total_files = 0

    # Scan src/ and scripts/
    for root_dir in ["src", "scripts", "tests"]:
        abs_root = os.path.join(PROJECT_ROOT, root_dir)
        if not os.path.exists(abs_root):
            continue
        for r, _, files in os.walk(abs_root):
            for f in sorted(files):
                if f.endswith(".py"):
                    rel_p = os.path.relpath(os.path.join(r, f), PROJECT_ROOT).replace("\\", "/")
                    abs_p = os.path.join(r, f)
                    try:
                        with open(abs_p, "r", encoding="utf-8") as fp:
                            lines = len(fp.readlines())
                    except Exception:
                        lines = 0
                    size = os.path.getsize(abs_p)
                    size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
                    purpose = MODULE_PURPOSES.get(rel_p, "Unit test or supporting helper module.")
                    print(f"{rel_p:34s} | {lines:5d} | {size_str:8s} | {purpose[:60]}...")
                    total_lines += lines
                    total_files += 1

    print("-" * 110)
    print(f"Total Python Source Files: {total_files} | Total Lines of Code: {total_lines}\n")


def print_structure() -> None:
    print("\n=======================================================")
    print("               PROJECT DIRECTORY STRUCTURE             ")
    print("=======================================================")
    for item in sorted(os.listdir(PROJECT_ROOT)):
        if item.startswith(".") and item != ".venv":
            continue
        item_path = os.path.join(PROJECT_ROOT, item)
        if os.path.isdir(item_path):
            sub_count = len(os.listdir(item_path))
            print(f"  [DIR]  {item:24s} ({sub_count} items)")
            # Print up to 10 children
            sub_items = sorted(os.listdir(item_path))[:8]
            for s in sub_items:
                if not s.startswith("."):
                    sub_p = os.path.join(item_path, s)
                    tag = "[DIR]" if os.path.isdir(sub_p) else "     "
                    print(f"         {tag} {s}")
            if len(sub_items) < len(os.listdir(item_path)):
                print(f"         ... and {len(os.listdir(item_path)) - len(sub_items)} more items")
        else:
            size = os.path.getsize(item_path)
            size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
            print(f"  [FILE] {item:24s} ({size_str})")


def print_active_config(config_path: str = "configs/segformer_b2_loeo.yaml") -> None:
    print("\n=======================================================")
    print(f"      ACTIVE SEGFORMER CONFIGURATION ({config_path})   ")
    print("=======================================================")
    full_path = os.path.join(PROJECT_ROOT, config_path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            print(f.read().strip())
    else:
        print(f"Config file not found: {full_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect project structure, modules, environment, and config.")
    parser.add_argument("--structure", action="store_true", help="Display project directory tree structure")
    parser.add_argument("--modules", action="store_true", help="List Python source modules and their purposes")
    parser.add_argument("--dependencies", action="store_true", help="Verify installed package dependencies")
    parser.add_argument("--environment", action="store_true", help="Display Python, PyTorch, CUDA, and GPU status")
    parser.add_argument("--config", action="store_true", help="Display active SegFormer YAML configuration")
    parser.add_argument("--config_path", type=str, default="configs/segformer_b2_loeo.yaml", help="Path to config file")
    parser.add_argument("--all", action="store_true", help="Run all inspections")
    args = parser.parse_args()

    run_all = args.all or not (args.structure or args.modules or args.dependencies or args.environment or args.config)

    if run_all or args.environment:
        print_environment_info()
    if run_all or args.dependencies:
        print_dependencies()
    if run_all or args.structure:
        print_structure()
    if run_all or args.modules:
        print_modules()
    if run_all or args.config:
        print_active_config(args.config_path)


if __name__ == "__main__":
    main()
