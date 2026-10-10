"""Unit tests for LOEO event splits and leakage prevention (Test 2 & Test 10)."""

import os
import json
import pytest

from src.data.splits import generate_loeo_splits
from src.models.segformer import SegFormerFloodModel


def test_loeo_event_disjointness_and_no_leakage():
    """Assert that for every fold, Train, Val, and Test events are strictly disjoint."""
    manifest_path = "outputs/dataset_manifest.csv"
    if not os.path.exists(manifest_path):
        pytest.skip("Dataset manifest not generated yet.")

    splits = generate_loeo_splits(
        dataset_manifest_path=manifest_path,
        splits_output_dir="outputs/splits",
    )

    assert len(splits) == 19, f"Expected 19 folds, got {len(splits)}"

    tested_events = set()
    for fold in splits:
        f_id = fold["fold_id"]
        test_e = fold["test_event"]
        train_es = set(fold["train_events"])
        val_es = set(fold["val_events"])

        # 1. Zero overlap assertions
        assert test_e not in train_es, f"Leakage in {f_id}: Test event '{test_e}' found in train events!"
        assert test_e not in val_es, f"Leakage in {f_id}: Test event '{test_e}' found in val events!"
        assert not (train_es & val_es), f"Leakage in {f_id}: Train and Val events overlap: {train_es & val_es}"

        # 2. Each event tested exactly once
        assert test_e not in tested_events, f"Event '{test_e}' tested multiple times!"
        tested_events.add(test_e)

    assert len(tested_events) == 19, "Not all 19 events were tested!"


def test_fold_isolation_fresh_initialization():
    """Verify that multiple model initializations create independent weight instances."""
    model_a = SegFormerFloodModel(pretrained=False)
    model_b = SegFormerFloodModel(pretrained=False)

    # Check that weights are independent objects and not sharing memory
    param_a = next(model_a.parameters())
    param_b = next(model_b.parameters())

    assert param_a.data_ptr() != param_b.data_ptr(), "Models share memory pointers!"
