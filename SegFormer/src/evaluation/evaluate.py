"""Evaluation Engine for SegFormer Flood Inundation Pipeline.

Performs deterministic patch evaluation, prediction reconstruction,
and whole-image confusion matrix accumulation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import FloodPlanetDataset
from src.data.preprocessing import reconstruct_full_image_logits
from src.evaluation.metrics import ConfusionMatrixAccumulator
from src.utils.logging import get_logger

logger = get_logger("Evaluation")


def evaluate_dataset(
    model: nn.Module,
    dataset: FloodPlanetDataset,
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 0,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluate model on a FloodPlanetDataset using tiled inference and full image reconstruction.
    
    Ensures that metrics are accumulated exactly once per original valid pixel.
    
    Args:
        model: SegFormer model.
        dataset: FloodPlanetDataset in 'val' or 'test' mode.
        batch_size: Batch size for tile inference.
        device: Torch device.
        num_workers: DataLoader worker count.
        
    Returns:
        Tuple of (metrics_dictionary, confusion_matrix).
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    accumulator = ConfusionMatrixAccumulator(num_classes=2, ignore_index=dataset.ignore_index)
    total_loss = 0.0
    loss_batches = 0

    # Store tile predictions grouped by sample_idx for reconstruction
    sample_tiles_preds: Dict[int, List[Tuple[np.ndarray, Tuple[int, int, int, int]]]] = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            sample_indices = batch["sample_idx"].tolist()
            coords = batch["coords"].tolist()

            outputs = model(images, labels=masks)
            if "loss" in outputs:
                total_loss += outputs["loss"].item()
                loss_batches += 1

            logits = outputs["logits"].cpu().numpy()  # (B, 2, H, W)

            for i in range(len(sample_indices)):
                s_idx = sample_indices[i]
                c_tuple = tuple(coords[i])
                if s_idx not in sample_tiles_preds:
                    sample_tiles_preds[s_idx] = []
                sample_tiles_preds[s_idx].append((logits[i], c_tuple))

    # Reconstruct each full original image and accumulate confusion matrix
    for s_idx, tile_data in sample_tiles_preds.items():
        sample_meta = dataset.get_full_sample(s_idx)
        full_shape = sample_meta["shape"]
        target_mask = sample_meta["target_mask"]

        tile_logits = [t[0] for t in tile_data]
        tile_coords = [t[1] for t in tile_data]

        reconstructed_logits = reconstruct_full_image_logits(
            tile_logits=tile_logits,
            tile_coords=tile_coords,
            full_shape=full_shape,
            num_classes=2,
        )

        pred_mask = np.argmax(reconstructed_logits, axis=0)  # (H, W)
        accumulator.update(pred_mask, target_mask)

    metrics = accumulator.compute_metrics()
    avg_loss = total_loss / max(1, loss_batches)
    metrics["eval_loss"] = round(avg_loss, 6)

    return metrics, accumulator.matrix
