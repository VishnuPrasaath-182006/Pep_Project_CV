"""Visualization Module for SegFormer Flood Inundation Experiments.

Generates:
- 4-Panel Qualitative Comparisons: (RGB, Ground Truth, Prediction, Color-Coded Error Map).
  Colors: TP=Green, TN=Black, FP=Red, FN=Blue, Ignore=Gray.
- Training Curves: Loss and Validation Flood IoU over 10 epochs.
- Cross-Validation Metrics Bar Charts (Event-wise IoU, Precision, Recall).
- Confusion Matrix Heatmaps.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger("Visualization")


def create_error_map(
    target: np.ndarray,
    pred: np.ndarray,
    ignore_index: int = 255,
) -> np.ndarray:
    """Create an RGB error map highlighting TP, TN, FP, FN, and Ignored pixels.
    
    Color codes:
      - True Positive (TP: target=1, pred=1): Green [0, 200, 0]
      - True Negative (TN: target=0, pred=0): Black / Dark Gray [30, 30, 30]
      - False Positive (FP: target=0, pred=1): Red [220, 20, 20]
      - False Negative (FN: target=1, pred=0): Blue [30, 100, 240]
      - Ignored / Nodata: Light Gray [180, 180, 180]
    """
    h, w = target.shape[:2]
    error_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Ignored pixels
    ign_mask = (target == ignore_index)
    error_rgb[ign_mask] = [180, 180, 180]

    valid = ~ign_mask
    tp = valid & (target == 1) & (pred == 1)
    tn = valid & (target == 0) & (pred == 0)
    fp = valid & (target == 0) & (pred == 1)
    fn = valid & (target == 1) & (pred == 0)

    error_rgb[tp] = [0, 200, 0]     # Green
    error_rgb[tn] = [30, 30, 30]    # Dark
    error_rgb[fp] = [220, 20, 20]   # Red
    error_rgb[fn] = [30, 100, 240]  # Blue

    return error_rgb


def save_prediction_panel(
    rgb_image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    save_path: str,
    event_name: str,
    chip_id: str,
    ignore_index: int = 255,
) -> None:
    """Render and save a 4-panel figure for a sample chip."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    error_map = create_error_map(ground_truth, prediction, ignore_index=ignore_index)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    
    # 1. RGB
    # Scale optical image to [0, 1] for display if uint8/uint16
    disp_rgb = rgb_image.astype(np.float32)
    disp_rgb = np.clip(disp_rgb / 255.0, 0.0, 1.0)
    axes[0].imshow(disp_rgb)
    axes[0].set_title(f"PlanetScope RGB\n({event_name} - {chip_id})", fontsize=11)
    axes[0].axis("off")

    # 2. Ground Truth
    gt_display = np.zeros_like(ground_truth, dtype=np.float32)
    gt_display[ground_truth == 1] = 1.0
    gt_display[ground_truth == ignore_index] = 0.5
    axes[1].imshow(ground_truth == 1, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask\n(Blue = Flood)", fontsize=11)
    axes[1].axis("off")

    # 3. SegFormer Prediction
    axes[2].imshow(prediction == 1, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("SegFormer-B2 Prediction\n(Blue = Flood)", fontsize=11)
    axes[2].axis("off")

    # 4. Color-coded Error Map
    axes[3].imshow(error_map)
    axes[3].set_title("Error Map", fontsize=11)
    axes[3].axis("off")

    # Legend for error map
    patches = [
        mpatches.Patch(color=(0/255, 200/255, 0/255), label="True Positive (Flood)"),
        mpatches.Patch(color=(30/255, 30/255, 30/255), label="True Negative (Land)"),
        mpatches.Patch(color=(220/255, 20/255, 20/255), label="False Positive (False Alarm)"),
        mpatches.Patch(color=(30/255, 100/255, 240/255), label="False Negative (Missed Flood)"),
    ]
    axes[3].legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved visualization panel to {save_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    fold_name: str,
) -> None:
    """Plot and save training/validation loss and validation Flood IoU curves."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

    # Loss axis
    color_train = "tab:blue"
    color_val = "tab:orange"
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", color=color_train, fontsize=11)
    line1 = ax1.plot(epochs, history["train_loss"], label="Train Loss", color=color_train, marker="o", linewidth=2)
    line2 = ax1.plot(epochs, history["val_loss"], label="Val Loss", color=color_val, marker="s", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color_train)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # IoU axis
    ax2 = ax1.twinx()
    color_iou = "tab:green"
    ax2.set_ylabel("Validation Flood IoU", color=color_iou, fontsize=11)
    line3 = ax2.plot(epochs, history["val_flood_iou"], label="Val Flood IoU", color=color_iou, marker="^", linewidth=2, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color_iou)

    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)

    plt.title(f"Training Progress - {fold_name} (10 Epochs)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_event_metrics_bar(
    events: List[str],
    metric_values: List[float],
    metric_name: str,
    save_path: str,
) -> None:
    """Plot event-wise bar chart for a metric."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    y_pos = np.arange(len(events))
    bars = ax.bar(y_pos, metric_values, color="royalblue", edgecolor="black", alpha=0.85)

    ax.set_xticks(y_pos)
    ax.set_xticklabels(events, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"Held-Out Event Generalization: {metric_name}", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Mean reference line
    if len(metric_values) > 0:
        mean_val = float(np.mean(metric_values))
        ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Macro Mean: {mean_val:.3f}")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
