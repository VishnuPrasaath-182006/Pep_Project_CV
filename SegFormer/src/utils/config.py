import os
import yaml
from typing import Any, Dict


def load_config(config_path: str = "configs/segformer_b2_loeo.yaml") -> Dict[str, Any]:
    """Load and validate the experiment configuration YAML.
    
    Automatically checks and adapts dataset paths for Kaggle environments:
    - /kaggle/input/floodplanet
    - /kaggle/input/floodplanet-inundation-dataset
    - Datasets
    - data/raw/FloodPlanet
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Path resolution logic for portability
    raw_root = config.get("data", {}).get("dataset_root", "Datasets")
    
    # Check common fallback locations if specified path doesn't exist
    possible_roots = [
        raw_root,
        os.path.join(os.getcwd(), raw_root),
        "Datasets",
        os.path.join(os.getcwd(), "Datasets"),
        "data/raw/FloodPlanet",
        "/kaggle/input/floodplanet",
        "/kaggle/input/floodplanet/Datasets",
        "/kaggle/input/floodplanet-inundation-dataset",
        "/kaggle/input/floodplanet-inundation-dataset/Datasets",
    ]
    
    found_root = None
    for p in possible_roots:
        if os.path.exists(p) and os.path.isdir(p):
            # Check if it has event subdirectories
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subdirs) > 0:
                found_root = p
                break
                
    if found_root:
        config["data"]["dataset_root"] = found_root

    # Ensure output directories are defined
    for p_key in ["output_dir", "checkpoints_dir", "splits_dir", "logs_dir", "results_dir", "visualizations_dir"]:
        if p_key in config.get("paths", {}):
            os.makedirs(config["paths"][p_key], exist_ok=True)

    return config
