import os
import yaml
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "train_config.yaml"

def load_config(config_path=None):
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# Fixed Event-Level Split Definition
TRAIN_EVENTS = [
    "Colombia", "Ghana", "Nigeria", "Paraguay", "Somalia", "Spain",
    "US-Alabama", "US-Arkansas", "US-Carolina", "US-Kansas", "US-Nebraska",
    "US-Oklahoma", "US-Texas", "Uzbekistan"
]

VAL_EVENTS = [
    "Bolivia", "US-Dakota"
]

TEST_EVENTS = [
    "Bangladesh", "Cambodia", "Nepal"
]

ALL_19_EVENTS = sorted(TRAIN_EVENTS + VAL_EVENTS + TEST_EVENTS)
