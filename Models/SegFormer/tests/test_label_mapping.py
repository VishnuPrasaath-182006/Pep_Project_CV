"""Unit tests for label mapping and ignored pixel handling (Test 3 & Test 8)."""

import numpy as np
import pytest
import torch

from src.data.label_processing import (
    map_labels,
    validate_label_tensor,
    RAW_NODATA,
    RAW_NON_FLOOD,
    RAW_FLOOD,
    TARGET_NON_FLOOD,
    TARGET_FLOOD,
    TARGET_IGNORE,
)


def test_label_mapping_numpy():
    """Verify exact raw-to-training label conversion for NumPy arrays."""
    raw = np.array([
        [RAW_NON_FLOOD, RAW_FLOOD],
        [RAW_NODATA, RAW_NON_FLOOD],
    ], dtype=np.uint8)

    target = map_labels(raw, ignore_index=255)
    assert target[0, 0] == TARGET_NON_FLOOD  # 0
    assert target[0, 1] == TARGET_FLOOD      # 1
    assert target[1, 0] == 255               # ignore
    assert target[1, 1] == TARGET_NON_FLOOD  # 0


def test_label_mapping_tensor():
    """Verify exact raw-to-training label conversion for PyTorch Tensors."""
    raw = torch.tensor([
        [1.0, 2.0],
        [0.0, 1.0],
    ], dtype=torch.float32)

    target = map_labels(raw, ignore_index=255)
    assert target[0, 0].item() == TARGET_NON_FLOOD
    assert target[0, 1].item() == TARGET_FLOOD
    assert target[1, 0].item() == 255
    assert target[1, 1].item() == TARGET_NON_FLOOD


def test_unexpected_raw_label_raises_error():
    """Ensure ValueError is raised if raw labels contain invalid values."""
    raw = np.array([[1, 2], [3, 1]], dtype=np.uint8)  # 3 is unexpected
    with pytest.raises(ValueError):
        map_labels(raw)


def test_validate_label_tensor():
    """Verify validation passes for valid tensors and fails on invalid class IDs."""
    valid_tensor = torch.tensor([[0, 1], [255, 0]], dtype=torch.long)
    assert validate_label_tensor(valid_tensor, num_classes=2, ignore_index=255)

    invalid_tensor = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    with pytest.raises(ValueError):
        validate_label_tensor(invalid_tensor, num_classes=2, ignore_index=255)
