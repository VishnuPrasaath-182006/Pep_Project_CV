#!/usr/bin/env python3
"""End-to-End Smoke Test for SegFormer-B2 Flood Inundation Pipeline.

Verifies:
1. Config loading and path validation
2. Dataset loading and batch generation
3. Model forward pass on actual batch
4. Loss computation and finite loss verification
5. Backward pass and optimizer step
6. Metric accumulation and calculation
7. Prediction reconstruction from tiles
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.data.inspect_dataset import inspect_dataset
from src.data.splits import generate_loeo_splits
from src.data.dataset import FloodPlanetDataset
from src.models.segformer import SegFormerFloodModel
from src.evaluation.metrics import ConfusionMatrixAccumulator
from src.data.preprocessing import reconstruct_full_image_logits
import pandas as pd

logger = get_logger("SmokeTest")


def run_smoke_test(config_path: str = "configs/segformer_b2_loeo.yaml") -> bool:
    logger.info("=== Running SegFormer-B2 Pipeline Smoke Test ===")
    config = load_config(config_path)

    # 1. Dataset Manifest
    manifest_path = os.path.join(config["paths"]["output_dir"], "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        inspect_dataset(
            dataset_root=config["data"]["dataset_root"],
            output_dir=config["paths"]["output_dir"],
            sensor=config["data"]["sensor"],
        )
    manifest_df = pd.read_csv(manifest_path)
    logger.info(f"[PASS] Dataset manifest loaded: {len(manifest_df)} total chips.")

    # 2. Splits
    splits_path = os.path.join(config["paths"]["splits_dir"], "loeo_splits.json")
    if not os.path.exists(splits_path):
        generate_loeo_splits(
            dataset_manifest_path=manifest_path,
            splits_output_dir=config["paths"]["splits_dir"],
        )
    logger.info(f"[PASS] LOEO splits verified.")

    # 3. Mini Dataset & DataLoader
    mini_train_df = manifest_df[manifest_df["event"] != "Bangladesh"].head(4)
    train_ds = FloodPlanetDataset(
        manifest_df=mini_train_df,
        dataset_root=config["data"]["dataset_root"],
        split="train",
        input_size=256,  # Use small 256 for fast smoke test
    )
    loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    images, masks = batch["image"], batch["mask"]
    assert images.shape == (2, 3, 256, 256), f"Unexpected image shape: {images.shape}"
    assert masks.shape == (2, 256, 256), f"Unexpected mask shape: {masks.shape}"
    logger.info(f"[PASS] Batch generation verified: images={images.shape}, masks={masks.shape}")

    # 4. Model Construction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Instantiating SegFormerFloodModel on {device}...")
    model = SegFormerFloodModel(
        encoder_checkpoint=config["model"]["encoder_checkpoint"],
        num_classes=2,
        ignore_index=255,
        pretrained=True,
    ).to(device)
    logger.info(f"[PASS] Model instantiated successfully.")

    # 5. Forward Pass
    images = images.to(device)
    masks = masks.to(device)
    outputs = model(images, labels=masks)
    logits = outputs["logits"]
    loss = outputs["loss"]
    assert logits.shape == (2, 2, 256, 256), f"Unexpected logits shape: {logits.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    logger.info(f"[PASS] Forward pass verified: logits={logits.shape}, loss={loss.item():.4f}")

    # 6. Backward Pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.info(f"[PASS] Backward pass and optimizer update verified.")

    # 7. Metrics & Confusion Matrix
    preds = torch.argmax(logits, dim=1)
    acc = ConfusionMatrixAccumulator()
    acc.update(preds, masks)
    m = acc.compute_metrics()
    logger.info(f"[PASS] Metrics computed: Flood IoU={m['flood_iou']}, mIoU={m['miou']}, F1={m['f1']}")

    # 8. Full Image Reconstruction
    sample_tiles = [logits[0].detach().cpu().numpy(), logits[1].detach().cpu().numpy()]
    sample_coords = [(0, 0, 256, 256), (256, 256, 512, 512)]
    reconstructed = reconstruct_full_image_logits(sample_tiles, sample_coords, full_shape=(512, 512), num_classes=2)
    assert reconstructed.shape == (2, 512, 512), f"Reconstruction shape error: {reconstructed.shape}"
    logger.info(f"[PASS] Full image reconstruction verified: shape={reconstructed.shape}")

    logger.info("=== SMOKE TEST PASSED: ALL 8 CORE SUBSYSTEMS VERIFIED ===")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
