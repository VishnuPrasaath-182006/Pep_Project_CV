import numpy as np
import torch

class MetricTracker:
    """
    Globally accumulated confusion matrix tracker for semantic segmentation metrics.
    Excludes pixels with IGNORE_INDEX (-1).
    """
    def __init__(self, num_classes=2, ignore_index=-1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        # Confusion matrix shape: (num_classes, num_classes)
        # Row: Ground Truth, Column: Prediction
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        """
        preds: torch.Tensor or np.ndarray of shape (B, H, W) or (B, C, H, W)
        targets: torch.Tensor or np.ndarray of shape (B, H, W)
        """
        if isinstance(preds, torch.Tensor):
            if preds.dim() == 4:
                preds = torch.argmax(preds, dim=1)
            preds = preds.detach().cpu().numpy()
        elif preds.ndim == 4:
            preds = np.argmax(preds, axis=1)
            
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        preds = preds.astype(np.int64)
        targets = targets.astype(np.int64)
        
        valid_mask = (targets != self.ignore_index)
        valid_preds = preds[valid_mask]
        valid_targets = targets[valid_mask]
        
        if len(valid_targets) == 0:
            return
            
        mask = (valid_targets >= 0) & (valid_targets < self.num_classes)
        cm = np.bincount(
            self.num_classes * valid_targets[mask] + valid_preds[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += cm

    def compute(self):
        cm = self.confusion_matrix.astype(np.float64)
        
        # TN, FP, FN, TP for binary classification (Class 0: BG, Class 1: Flood)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # IoU per class
        iou = tp / np.maximum(1.0, tp + fp + fn)
        bg_iou = float(iou[0])
        flood_iou = float(iou[1]) if self.num_classes > 1 else float(iou[0])
        mean_iou = float(np.mean(iou))
        
        # Precision, Recall, F1 / Dice per class
        precision = tp / np.maximum(1.0, tp + fp)
        recall = tp / np.maximum(1.0, tp + fn)
        dice = (2.0 * tp) / np.maximum(1.0, 2.0 * tp + fp + fn)
        
        bg_dice = float(dice[0])
        flood_dice = float(dice[1]) if self.num_classes > 1 else float(dice[0])
        flood_precision = float(precision[1]) if self.num_classes > 1 else float(precision[0])
        flood_recall = float(recall[1]) if self.num_classes > 1 else float(recall[0])
        
        # Overall Pixel Accuracy
        total_pixels = cm.sum()
        accuracy = float(np.trace(cm) / max(1.0, total_pixels))
        
        return {
            "bg_iou": bg_iou,
            "flood_iou": flood_iou,
            "mean_iou": mean_iou,
            "flood_dice": flood_dice,
            "flood_precision": flood_precision,
            "flood_recall": flood_recall,
            "pixel_accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "tp": int(tp[1]) if len(tp) > 1 else int(tp[0]),
            "tn": int(tn[1]) if len(tn) > 1 else int(tn[0]),
            "fp": int(fp[1]) if len(fp) > 1 else int(fp[0]),
            "fn": int(fn[1]) if len(fn) > 1 else int(fn[0])
        }
