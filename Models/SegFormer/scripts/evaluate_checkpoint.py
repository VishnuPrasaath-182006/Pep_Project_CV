#!/usr/bin/env python3
"""Checkpoint Evaluation CLI Tool for SegFormer-B2 LOEO-CV Pipeline.

Evaluates a saved SegFormer checkpoint strictly on its held-out test event.
- Verifies that the test event is completely unseen in training and validation.
- Reconstructs full 1024x1024 predictions from 512x512 tiles.
- Accumulates the whole-image confusion matrix.
- Computes: Flood IoU, Non-Flood IoU, mIoU, Dice/F1, Precision, Recall, Specificity, Accuracy.
- Generates 4-panel prediction visualizations with TP, TN, FP, FN error maps.
- Optionally opens visualization with --open.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import FloodPlanetDataset
from src.data.preprocessing import reconstruct_full_image_logits
from src.evaluation.metrics import ConfusionMatrixAccumulator
from src.models.segformer import build_segformer_model
from src.utils.config import load_config
from src.visualization.visualize import save_prediction_panel


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str = "configs/segformer_b2_loeo.yaml",
    fold_idx: int = 0,
    open_viz: bool = False,
) -> Dict[str, Any]:
    config = load_config(config_path)
    split_path = os.path.join(PROJECT_ROOT, "outputs", "splits", f"fold_{fold_idx:02d}.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    test_event = split_data["test_event"]
    train_events = split_data["train_events"]
    val_events = split_data["val_events"]

    manifest_path = os.path.join(PROJECT_ROOT, "outputs", "dataset_manifest.csv")
    manifest_df = pd.read_csv(manifest_path)
    test_chips = manifest_df[manifest_df["event"] == test_event].to_dict(orient="records")

    # Strict isolation assertion
    assert test_event not in train_events, f"CRITICAL LEAKAGE: Test event {test_event} found in train events!"
    assert test_event not in val_events, f"CRITICAL LEAKAGE: Test event {test_event} found in val events!"

    print("\n=======================================================")
    print("        SEGFORMER HELD-OUT CHECKPOINT EVALUATION       ")
    print("=======================================================")
    print(f"Checkpoint Path    : {checkpoint_path}")
    print(f"Fold ID            : Fold {fold_idx:02d}")
    print(f"Held-Out Event     : {test_event} (STRICTLY UNSEEN)")
    print(f"Test Chips Count   : {len(test_chips)}")
    print("-------------------------------------------------------")

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segformer_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        ignore_index=config["model"]["ignore_index"],
        pretrained_checkpoint=config["model"]["encoder_checkpoint"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_df = manifest_df[manifest_df["event"] == test_event].reset_index(drop=True)
    test_dataset = FloodPlanetDataset(
        manifest_df=test_df,
        dataset_root=config["data"]["dataset_root"],
        split="test",
        input_size=config["data"]["input_size"],
        rgb_bands=tuple(config["data"].get("rgb_bands", [2, 1, 0])),
        image_mean=tuple(config["data"].get("image_mean", [0.485, 0.456, 0.406])),
        image_std=tuple(config["data"].get("image_std", [0.229, 0.224, 0.225])),
        ignore_index=config["model"]["ignore_index"],
    )

    from src.evaluation.evaluate import evaluate_dataset
    from src.data.preprocessing import normalize_image

    print("Evaluating model on held-out test dataset...")
    metrics, cm = evaluate_dataset(
        model=model,
        dataset=test_dataset,
        batch_size=config["training"].get("batch_size", 4),
        device=device,
        num_workers=0,
    )

    saved_viz_paths = []
    num_samples_to_viz = min(3, len(test_df))
    for s_idx in range(num_samples_to_viz):
        full_sample = test_dataset.get_full_sample(s_idx)
        rgb_img = full_sample["rgb_image"]
        target_mask = full_sample["target_mask"]
        chip_id = full_sample["filename"]
        tiles = full_sample["tiles"]
        full_shape = full_sample["shape"]

        tile_logits = []
        tile_coords = []
        for ymin, xmin, ymax, xmax in tiles:
            crop_img = rgb_img[ymin:ymax, xmin:xmax]
            pad_h = config["data"]["input_size"] - crop_img.shape[0]
            pad_w = config["data"]["input_size"] - crop_img.shape[1]
            if pad_h > 0 or pad_w > 0:
                crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            
            norm_img = normalize_image(
                crop_img,
                mean=tuple(config["data"].get("image_mean", (0.485, 0.456, 0.406))),
                std=tuple(config["data"].get("image_std", (0.229, 0.224, 0.225))),
            )
            tensor_crop = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                out = model(tensor_crop)
                logit_tile = out["logits"].squeeze(0).cpu().numpy()
            tile_logits.append(logit_tile)
            tile_coords.append((ymin, xmin, ymax, xmax))

        full_logits = reconstruct_full_image_logits(tile_logits, tile_coords, full_shape, num_classes=2)
        pred_mask = np.argmax(full_logits, axis=0)

        viz_out = os.path.join(PROJECT_ROOT, "outputs", "visualizations", test_event, f"{chip_id}_panel.png")
        save_prediction_panel(
            rgb_image=rgb_img,
            ground_truth=target_mask,
            prediction=pred_mask,
            save_path=viz_out,
            event_name=test_event,
            chip_id=chip_id,
            ignore_index=config["model"]["ignore_index"],
        )
        saved_viz_paths.append(viz_out)

    # Confusion matrix values: class 0 = Land, class 1 = Flood
    # cm[gt, pred]
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    print("\n-------------------------------------------------------")
    print("                  CONFUSION MATRIX                     ")
    print("-------------------------------------------------------")
    print(f"  True Positives  (TP - Flood correctly identified) : {tp:10,d}")
    print(f"  True Negatives  (TN - Land correctly identified)  : {tn:10,d}")
    print(f"  False Positives (FP - False Alarm)                : {fp:10,d}")
    print(f"  False Negatives (FN - Missed Flood)               : {fn:10,d}")
    print("-------------------------------------------------------")
    print("                EVALUATION METRICS                     ")
    print("-------------------------------------------------------")
    print(f"  Flood IoU     : {metrics['flood_iou']:.4f}")
    print(f"  Non-Flood IoU : {metrics['non_flood_iou']:.4f}")
    print(f"  mIoU          : {metrics['miou']:.4f}")
    print(f"  Flood F1/Dice : {metrics['dice']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  Specificity   : {metrics['specificity']:.4f}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print("=======================================================\n")

    # Save metrics JSON
    metrics_out = os.path.join(PROJECT_ROOT, "outputs", "results", f"fold_{fold_idx:02d}_test_metrics.json")
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump({
            "fold_idx": fold_idx,
            "test_event": test_event,
            "metrics": metrics,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_out}")

    if open_viz and saved_viz_paths:
        target_to_open = saved_viz_paths[0]
        print(f"Opening evaluation figure: {target_to_open}")
        if sys.platform == "win32":
            os.system(f'start "" "{target_to_open}"')

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved SegFormer checkpoint on its held-out test event.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/fold_00_best.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/segformer_b2_loeo.yaml", help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0 to 18)")
    parser.add_argument("--open", action="store_true", help="Open generated prediction figure")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        fold_idx=args.fold,
        open_viz=args.open,
    )


if __name__ == "__main__":
    main()
