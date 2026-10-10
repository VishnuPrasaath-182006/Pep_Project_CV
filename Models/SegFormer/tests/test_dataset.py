"""Unit tests for dataset loading, pairing, and tensor shapes (Test 1 & Test 4)."""

import os
import pandas as pd
import pytest
import torch

from src.data.dataset import FloodPlanetDataset


@pytest.fixture
def manifest_df():
    manifest_path = "outputs/dataset_manifest.csv"
    if not os.path.exists(manifest_path):
        pytest.skip("Dataset manifest not generated yet.")
    return pd.read_csv(manifest_path)


def test_dataset_pairing_and_integrity(manifest_df):
    """Confirm that all 366 images and masks are paired with valid paths."""
    assert len(manifest_df) == 366
    for _, row in manifest_df.iterrows():
        img_full = os.path.join("Datasets", row["image_path"])
        mask_full = os.path.join("Datasets", row["mask_path"])
        assert os.path.exists(img_full), f"Missing image: {img_full}"
        assert os.path.exists(mask_full), f"Missing mask: {mask_full}"
        assert row["width"] == 1024
        assert row["height"] == 1024
        assert row["bands"] == 4


def test_dataset_tensor_shapes(manifest_df):
    """Validate image tensors [3, 512, 512] and mask tensors [512, 512]."""
    dataset = FloodPlanetDataset(
        manifest_df=manifest_df.head(2),
        dataset_root="Datasets",
        split="train",
        input_size=512,
    )

    sample = dataset[0]
    img = sample["image"]
    mask = sample["mask"]

    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.shape == (3, 512, 512), f"Expected (3, 512, 512), got {img.shape}"
    assert mask.shape == (512, 512), f"Expected (512, 512), got {mask.shape}"
    assert img.dtype == torch.float32
    assert mask.dtype == torch.long
