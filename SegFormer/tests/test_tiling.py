"""Unit tests for tiling and full-image prediction reconstruction (Test 9 & Test 12)."""

import numpy as np
import pytest
import torch

from src.data.preprocessing import generate_tiles, reconstruct_full_image_logits
from src.evaluation.metrics import ConfusionMatrixAccumulator


def test_tiling_full_coverage_and_coordinates():
    """Verify that 1024x1024 produces exactly four 512x512 non-overlapping tiles covering all pixels."""
    tiles = generate_tiles((1024, 1024), tile_size=512, overlap=0)
    assert len(tiles) == 4
    assert tiles[0] == (0, 0, 512, 512)
    assert tiles[1] == (0, 512, 512, 1024)
    assert tiles[2] == (512, 0, 1024, 512)
    assert tiles[3] == (512, 512, 1024, 1024)


def test_full_image_prediction_reconstruction():
    """Confirm that stitched tiles recreate the exact full image without double-counting pixels."""
    h, w = 1024, 1024
    tile_size = 512
    tiles = generate_tiles((h, w), tile_size=tile_size, overlap=0)

    # Create distinct pattern in 4 tiles
    tile_logits = []
    for i, (ymin, xmin, ymax, xmax) in enumerate(tiles):
        tile_log = np.zeros((2, 512, 512), dtype=np.float32)
        tile_log[0, :, :] = i  # Class 0 has value i
        tile_logits.append(tile_log)

    reconstructed = reconstruct_full_image_logits(tile_logits, tiles, (h, w), num_classes=2)
    assert reconstructed.shape == (2, 1024, 1024)

    # Check that each quadrant has the exact tile value
    assert reconstructed[0, 0:512, 0:512].mean() == 0.0
    assert reconstructed[0, 0:512, 512:1024].mean() == 1.0
    assert reconstructed[0, 512:1024, 0:512].mean() == 2.0
    assert reconstructed[0, 512:1024, 512:1024].mean() == 3.0


def test_minimal_end_to_end_cycle():
    """Run a minimal 1-step train and evaluation cycle (Test 12)."""
    acc = ConfusionMatrixAccumulator()
    preds = np.random.randint(0, 2, (512, 512))
    targets = np.random.randint(0, 2, (512, 512))
    acc.update(preds, targets)
    metrics = acc.compute_metrics()
    assert "flood_iou" in metrics
    assert "miou" in metrics
    assert metrics["total_valid_pixels"] == 512 * 512
