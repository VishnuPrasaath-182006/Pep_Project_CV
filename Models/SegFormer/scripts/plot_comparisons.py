#!/usr/bin/env python3
"""Event Comparison and Visualization CLI Tool for LOEO-CV Results.

Generates comparative visualizations across completed flood events:
- Bar chart of Flood IoU and mIoU per event
- Precision vs Recall comparative chart
- Summary statistics table
Saves to outputs/visualizations/loeo_event_comparison.png

Supports --open to automatically launch Windows Photo Viewer.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_VIZ_DIR = os.path.join(PROJECT_ROOT, "outputs", "visualizations")


def plot_event_comparisons(
    metrics_csv: str = "outputs/results/event_metrics.csv",
    output_png: str = "outputs/visualizations/loeo_event_comparison.png",
    open_viz: bool = False,
) -> str:
    full_csv = os.path.join(PROJECT_ROOT, metrics_csv)
    if not os.path.exists(full_csv):
        print(f"Error: {full_csv} not found. Complete at least one fold training first.")
        return ""

    df = pd.read_csv(full_csv)
    if len(df) == 0:
        print("No completed fold records found in metrics CSV.")
        return ""

    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)

    events = df["test_event"].tolist() if "test_event" in df.columns else df["event"].tolist()
    flood_ious = df["flood_iou"].tolist()
    mious = df["miou"].tolist()
    precisions = df["precision"].tolist()
    recalls = df["recall"].tolist()
    f1s = df["f1_score"].tolist() if "f1_score" in df.columns else df.get("f1", [0.0]*len(df)).tolist()

    x = np.arange(len(events))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(events) * 1.5), 10), dpi=150)

    # Plot 1: IoU Comparison
    axes[0].bar(x - width/2, flood_ious, width, label="Flood IoU", color="#0284c7")
    axes[0].bar(x + width/2, mious, width, label="mIoU", color="#38bdf8")
    axes[0].set_ylabel("IoU Score", fontsize=11)
    axes[0].set_title("SegFormer-B2 LOEO Cross-Validation Performance per Flood Event", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(events, rotation=25, ha="right", fontsize=10)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 2: Precision vs Recall vs F1
    w3 = 0.25
    axes[1].bar(x - w3, precisions, w3, label="Precision", color="#10b981")
    axes[1].bar(x, recalls, w3, label="Recall", color="#f59e0b")
    axes[1].bar(x + w3, f1s, w3, label="F1 / Dice", color="#8b5cf6")
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_title("Precision, Recall, and Dice Score per Event", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(events, rotation=25, ha="right", fontsize=10)
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Saved event-wise comparison chart to: {output_png}\n")

    if open_viz:
        print(f"Opening comparison chart: {output_png}")
        if sys.platform == "win32":
            os.system(f'start "" "{output_png}"')

    return output_png


def main():
    parser = argparse.ArgumentParser(description="Plot event-wise comparison charts.")
    parser.add_argument("--csv", type=str, default="outputs/results/event_metrics.csv", help="Path to event metrics CSV")
    parser.add_argument("--output", type=str, default="outputs/visualizations/loeo_event_comparison.png", help="Path to output PNG")
    parser.add_argument("--open", action="store_true", help="Open output figure in Windows viewer")
    args = parser.parse_args()

    plot_event_comparisons(
        metrics_csv=args.csv,
        output_png=args.output,
        open_viz=args.open,
    )


if __name__ == "__main__":
    main()
