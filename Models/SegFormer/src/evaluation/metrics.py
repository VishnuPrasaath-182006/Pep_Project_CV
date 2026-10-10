"""Evaluation Metrics Module for Flood Inundation Semantic Segmentation.

Implements discrete integer confusion-matrix accumulation and standard remote sensing metrics:
- Flood IoU, Non-flood IoU, mIoU
- Flood Precision, Recall, F1 / Dice
- Specificity, Overall Accuracy
- Strict exclusion of ignore_index (255) pixels.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


class ConfusionMatrixAccumulator:
    """Accumulates integer confusion matrix over patches or full images without batch averaging."""

    def __init__(self, num_classes: int = 2, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Matrix layout: cm[target, pred]
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Reset the accumulator."""
        self.matrix.fill(0)

    def update(
        self,
        preds: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Accumulate predictions and ground-truth targets into the confusion matrix.
        
        Args:
            preds: Array/Tensor of predicted class IDs (0 or 1).
            targets: Array/Tensor of ground-truth class IDs (0, 1, or ignore_index).
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds = preds.ravel()
        targets = targets.ravel()

        # Mask out ignored/nodata pixels
        valid_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        v_preds = preds[valid_mask]
        v_targets = targets[valid_mask]

        # Filter out-of-bound predictions if any
        pred_valid = (v_preds >= 0) & (v_preds < self.num_classes)
        v_preds = v_preds[pred_valid]
        v_targets = v_targets[pred_valid]

        if len(v_targets) == 0:
            return

        # Fast bincount accumulation
        indices = v_targets * self.num_classes + v_preds
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all benchmark metrics from the accumulated confusion matrix.
        
        Matrix conventions:
          target 0 (Non-flood), pred 0 -> TN
          target 0 (Non-flood), pred 1 -> FP
          target 1 (Flood),     pred 0 -> FN
          target 1 (Flood),     pred 1 -> TP
        """
        cm = self.matrix
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])

        total_valid = tp + tn + fp + fn
        total_actual_flood = tp + fn
        total_pred_flood = tp + fp
        total_actual_non_flood = tn + fp

        # Flood IoU: TP / (TP + FP + FN)
        flood_denom = tp + fp + fn
        flood_iou = float(tp / flood_denom) if flood_denom > 0 else 0.0

        # Non-flood IoU: TN / (TN + FP + FN)
        non_flood_denom = tn + fp + fn
        non_flood_iou = float(tn / non_flood_denom) if non_flood_denom > 0 else 0.0

        # mIoU: mean(Flood IoU, Non-flood IoU)
        miou = float((flood_iou + non_flood_iou) / 2.0)

        # Precision: TP / (TP + FP)
        precision = float(tp / total_pred_flood) if total_pred_flood > 0 else 0.0

        # Recall (Sensitivity): TP / (TP + FN)
        recall = float(tp / total_actual_flood) if total_actual_flood > 0 else 0.0

        # F1 / Dice: 2 * TP / (2*TP + FP + FN)
        f1_denom = 2 * tp + fp + fn
        f1 = float(2 * tp / f1_denom) if f1_denom > 0 else 0.0

        # Specificity: TN / (TN + FP)
        specificity = float(tn / total_actual_non_flood) if total_actual_non_flood > 0 else 0.0

        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        accuracy = float((tp + tn) / total_valid) if total_valid > 0 else 0.0

        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total_valid_pixels": total_valid,
            "flood_iou": round(flood_iou, 6),
            "non_flood_iou": round(non_flood_iou, 6),
            "miou": round(miou, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "dice": round(f1, 6),
            "specificity": round(specificity, 6),
            "accuracy": round(accuracy, 6),
            "confusion_matrix": cm.tolist(),
        }
