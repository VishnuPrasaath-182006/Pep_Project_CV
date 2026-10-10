"""
CAVE-Flood Deployment Configuration & Path Resolver
Maintains official hyperparameters, cryptographic checksums, and execution device setup.
"""

import os
import hashlib
from pathlib import Path
import torch

DEPLOYMENT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = DEPLOYMENT_DIR / "checkpoints"
ASSETS_DIR = DEPLOYMENT_DIR / "assets"
EXAMPLES_DIR = DEPLOYMENT_DIR / "examples"

# Official PlanetScope Normalization Parameters (computed from training rasters)
# Bands: [Red, Green, Blue, NIR]
NORM_MEAN = [2334.81, 2378.43, 2185.12, 3608.97]
NORM_STD = [1245.52, 1184.21, 1148.65, 1420.33]

# Candidate Region Selection Hyperparameters
PATCH_SIZE = 512
REGION_SIZE = 128
TOP_K = 4
MIN_COVERAGE = 0.5
LAMBDA_RES = 1.0

# Base UNet-Mamba Model Architecture Hyperparameters
MODEL_CONFIG = {
    "model": {
        "name": "UNetMamba",
        "in_channels": 4,
        "num_classes": 2,
        "embed_dim": 64,
        "decode_channels": 64,
        "pretrained": False,
    }
}

# Cryptographic Checksum Reference Ledger (SHA-256)
EXPECTED_HASHES = {
    "best_model.pth": "58674c84e97e16c7ab86f2e218ba8738c7aaf69a8c6fd8c384de00beec9eb76d",
    "best_e1_model.pth": "84a02d1f1fe8371bf799edb0ddce28f69d20dd90b3da503c06eac87f521226e9",
    "best_e2_model.pth": "19347f5f64385c46f16c27b345a7f143e56b51f23c1fe7e887906b6e7c19d2d9",
}

def get_device():
    """Detect and return optimal compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def compute_sha256(file_path):
    """Calculates SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

HF_WEIGHTS_REPO = "VP182006/CAVE-Flood-Weights"

def ensure_checkpoint_exists(filename):
    """Ensures checkpoint file is present locally; auto-downloads from HF Hub if missing."""
    p = CHECKPOINTS_DIR / filename
    if not p.exists():
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            print(f"[*] Downloading {filename} from {HF_WEIGHTS_REPO}...")
            downloaded = hf_hub_download(
                repo_id=HF_WEIGHTS_REPO,
                filename=filename,
                local_dir=str(CHECKPOINTS_DIR)
            )
            return Path(downloaded)
        except Exception as e:
            print(f"[!] Warning: Could not download {filename} from Hub: {e}")
    return p

def verify_checkpoints():
    """
    Verifies existence and cryptographic integrity of required model checkpoints.
    Returns: dict of model statuses and warnings.
    """
    status = {}
    for filename, expected_sha in EXPECTED_HASHES.items():
        p = ensure_checkpoint_exists(filename)
        if not p.exists():
            status[filename] = {"found": False, "verified": False, "error": f"Missing: {p}"}
        else:
            actual_sha = compute_sha256(p)
            is_valid = (actual_sha == expected_sha)
            status[filename] = {
                "found": True,
                "verified": is_valid,
                "path": str(p),
                "sha256": actual_sha,
                "expected_sha": expected_sha
            }
            
    scaler_path = ensure_checkpoint_exists("remoteclip_score_scaler.json")
    status["scaler"] = {
        "found": scaler_path.exists(),
        "path": str(scaler_path) if scaler_path.exists() else None
    }
    return status

