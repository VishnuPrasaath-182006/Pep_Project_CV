import os
import sys
import torch
import numpy as np
import pandas as pd

from config import load_config, BASE_DIR, ALL_19_EVENTS
from utils import get_logger, get_device_info
from model import build_model
from losses import build_loss
from dataset import get_dataloaders
from metrics import MetricTracker

def verify_pipeline():
    logger = get_logger("PipelineVerifier")
    logger.info("Starting Automated Pipeline Verification...")
    
    # 1. Environment Verification
    dev_info = get_device_info()
    logger.info(f"Environment Info: OS={dev_info['os']}, Python={dev_info['python']}, PyTorch={dev_info['pytorch']}, CUDA={dev_info['cuda_version']}")
    assert dev_info['cuda_available'], "CUDA GPU is required for training!"
    logger.info(f"Verified GPU: {dev_info['gpu_name']} ({dev_info['vram_gb']} GB VRAM, CC {dev_info['compute_capability']})")
    
    # 2. Config Verification
    config = load_config()
    assert config['data']['in_channels'] == 4, "IN_CHANNELS must be 4!"
    assert config['data']['num_classes'] == 2, "NUM_CLASSES must be 2!"
    assert config['data']['patch_size'] == 512, "PATCH_SIZE must be 512!"
    assert config['data']['stride'] in [256, 512], "STRIDE must be 256 or 512!"
    assert config['training']['epochs'] in [11, 21], "EPOCHS must be 11 or 21!"
    logger.info("Configuration parameters verified successfully.")
    
    # 3. Manifest Verification
    manifests_dir = BASE_DIR / config['data']['manifests_dir']
    train_manifest = manifests_dir / "train.csv"
    val_manifest = manifests_dir / "val.csv"
    test_manifest = manifests_dir / "test.csv"
    
    assert train_manifest.exists() and val_manifest.exists() and test_manifest.exists(), "Manifest files missing!"
    
    df_tr = pd.read_csv(train_manifest)
    df_va = pd.read_csv(val_manifest)
    df_te = pd.read_csv(test_manifest)
    
    events_tr = set(df_tr["event"].unique())
    events_va = set(df_va["event"].unique())
    events_te = set(df_te["event"].unique())
    
    assert events_tr.isdisjoint(events_va), "Train and Val events overlap!"
    assert events_tr.isdisjoint(events_te), "Train and Test events overlap!"
    assert events_va.isdisjoint(events_te), "Val and Test events overlap!"
    
    all_events = events_tr | events_va | events_te
    assert all_events == set(ALL_19_EVENTS), "All 19 events must be accounted for!"
    logger.info(f"Manifest verification passed: 14 Train, 2 Val, 3 Test events disjointly assigned.")
    
    # 4. Model Forward & Backward Pass Smoke Test
    device = torch.device("cuda")
    model = build_model(config).to(device)
    model.train()
    
    dummy_input = torch.randn(4, 4, 512, 512, device=device)
    dummy_target = torch.randint(0, 2, (4, 512, 512), device=device, dtype=torch.long)
    
    with torch.amp.autocast('cuda'):
        main_out, aux_out = model(dummy_input)
        assert main_out.shape == (4, 2, 512, 512), f"Main output shape mismatch: {main_out.shape}"
        assert aux_out.shape == (4, 2, 512, 512), f"Aux output shape mismatch: {aux_out.shape}"
        
        criterion = build_loss(config)
        loss = criterion((main_out, aux_out), dummy_target)
    loss.backward()
    
    logger.info(f"GPU Model Forward & Backward smoke test passed! Loss = {loss.item():.4f}")
    
    # 5. Metric Calculation Test
    tracker = MetricTracker(num_classes=2)
    tracker.update(main_out, dummy_target)
    metrics = tracker.compute()
    assert "mean_iou" in metrics and "flood_iou" in metrics, "Metric calculation missing required keys!"
    logger.info(f"Metric tracking verified successfully (Sample mIoU = {metrics['mean_iou']:.4f}).")
    
    logger.info("==========================================")
    logger.info("ALL PIPELINE VERIFICATION CHECKS PASSED!")
    logger.info("==========================================")
    return True

if __name__ == "__main__":
    verify_pipeline()
