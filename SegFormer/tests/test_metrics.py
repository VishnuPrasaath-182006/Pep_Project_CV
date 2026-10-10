"""Unit tests for metric calculation and confusion matrix accumulation (Test 7 & Test 8)."""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import ConfusionMatrixAccumulator


def test_manually_constructed_confusion_matrix():
    """Verify all metric formulas using a known manually constructed confusion matrix.
    
    Given:
      TN = 60, FP = 10, FN = 5, TP = 25
      Total valid = 100
      Total actual flood = TP + FN = 30
      Total predicted flood = TP + FP = 35
      Total actual non-flood = TN + FP = 70
      
    Formulas:
      Flood IoU: TP / (TP + FP + FN) = 25 / (25 + 10 + 5) = 25 / 40 = 0.625
      Non-flood IoU: TN / (TN + FP + FN) = 60 / (60 + 10 + 5) = 60 / 75 = 0.8
      mIoU: (0.625 + 0.8) / 2 = 0.7125
      Precision: TP / (TP + FP) = 25 / 35 ≈ 0.714286
      Recall: TP / (TP + FN) = 25 / 30 ≈ 0.833333
      F1: 2*TP / (2*TP + FP + FN) = 50 / (50 + 10 + 5) = 50 / 65 ≈ 0.769231
      Specificity: TN / (TN + FP) = 60 / 70 ≈ 0.857143
      Accuracy: (TP + TN) / 100 = 85 / 100 = 0.85
    """
    acc = ConfusionMatrixAccumulator(num_classes=2, ignore_index=255)
    # Target: 0 (non-flood), Pred: 0 -> TN
    # Target: 0, Pred: 1 -> FP
    # Target: 1 (flood), Pred: 0 -> FN
    # Target: 1, Pred: 1 -> TP
    targets = np.array([0]*70 + [1]*30)
    preds = np.array([0]*60 + [1]*10 + [0]*5 + [1]*25)

    acc.update(preds, targets)
    m = acc.compute_metrics()

    assert m["tn"] == 60
    assert m["fp"] == 10
    assert m["fn"] == 5
    assert m["tp"] == 25
    assert m["total_valid_pixels"] == 100

    assert pytest.approx(m["flood_iou"], abs=1e-5) == 0.625
    assert pytest.approx(m["non_flood_iou"], abs=1e-5) == 0.8
    assert pytest.approx(m["miou"], abs=1e-5) == 0.7125
    assert pytest.approx(m["precision"], abs=1e-5) == 25.0 / 35.0
    assert pytest.approx(m["recall"], abs=1e-5) == 25.0 / 30.0
    assert pytest.approx(m["f1"], abs=1e-5) == 50.0 / 65.0
    assert pytest.approx(m["specificity"], abs=1e-5) == 60.0 / 70.0
    assert pytest.approx(m["accuracy"], abs=1e-5) == 0.85


def test_ignored_pixels_excluded_from_metrics():
    """Verify that nodata / ignore_index=255 pixels have strictly zero effect on metrics."""
    acc_clean = ConfusionMatrixAccumulator(num_classes=2, ignore_index=255)
    targets_clean = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 0, 1])
    acc_clean.update(preds, targets_clean)
    metrics_clean = acc_clean.compute_metrics()

    # Now add ignored pixels (both where pred is 0 and 1)
    acc_dirty = ConfusionMatrixAccumulator(num_classes=2, ignore_index=255)
    targets_dirty = np.array([0, 0, 1, 1, 255, 255, 255])
    preds_dirty = np.array([0, 1, 0, 1, 0, 1, 0])
    acc_dirty.update(preds_dirty, targets_dirty)
    metrics_dirty = acc_dirty.compute_metrics()

    # The computed metrics and confusion counts must be completely identical
    assert metrics_dirty["tp"] == metrics_clean["tp"]
    assert metrics_dirty["tn"] == metrics_clean["tn"]
    assert metrics_dirty["fp"] == metrics_clean["fp"]
    assert metrics_dirty["fn"] == metrics_clean["fn"]
    assert metrics_dirty["flood_iou"] == metrics_clean["flood_iou"]
    assert metrics_dirty["miou"] == metrics_clean["miou"]
    assert metrics_dirty["accuracy"] == metrics_clean["accuracy"]
