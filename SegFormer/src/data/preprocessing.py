"""Preprocessing, Tiling, and Reconstruction Module for FloodPlanet Imagery.

Handles:
- PlanetScope 4-band to 3-band RGB extraction.
- Radiometric scaling from [0, 255] to [0.0, 1.0] and ImageNet mean/std normalization.
- Deterministic 512x512 tiling with coordinate bookkeeping.
- Seamless reconstruction of tiled predictions back to original full image coordinates.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch


DEFAULT_RGB_BANDS = (2, 1, 0)  # Red=2, Green=1, Blue=0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_rgb(image: np.ndarray, rgb_bands: Tuple[int, int, int] = DEFAULT_RGB_BANDS) -> np.ndarray:
    """Extract 3-channel optical RGB representation from multi-band imagery.
    
    Args:
        image: Array of shape (H, W, C) where C >= 3.
        rgb_bands: Tuple of band indices for (R, G, B).
        
    Returns:
        RGB array of shape (H, W, 3).
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array (H, W, C), got ndim={image.ndim} with shape {image.shape}")
    
    num_channels = image.shape[2]
    for b in rgb_bands:
        if b < 0 or b >= num_channels:
            raise IndexError(f"Band index {b} out of range for image with {num_channels} channels.")
            
    rgb = image[:, :, list(rgb_bands)]
    return rgb


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Scale imagery from [0, 255] to [0.0, 1.0] and apply ImageNet normalization.
    
    Args:
        image: RGB array (H, W, 3) with numeric values in range [0, 255].
        mean: Channel means.
        std: Channel standard deviations.
        
    Returns:
        Normalized float32 array of shape (H, W, 3).
    """
    # Cast to float32
    img_f32 = image.astype(np.float32)
    
    # Scale to [0.0, 1.0]
    img_scaled = img_f32 / 255.0
    
    # Standardize
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (img_scaled - mean_arr) / std_arr
    return normalized


def denormalize_image(
    tensor_or_arr: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Invert ImageNet normalization and scale back to [0, 255] uint8 for visualization."""
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img = (tensor_or_arr * std_arr) + mean_arr
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def generate_tiles(
    image_shape: Tuple[int, int],
    tile_size: int = 512,
    overlap: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Generate deterministic bounding box tiles covering the full image.
    
    Args:
        image_shape: (height, width) of the original raster.
        tile_size: Height and width of the square crop (e.g. 512).
        overlap: Pixel overlap between adjacent tiles (0 for exact non-overlapping grid).
        
    Returns:
        List of bounding boxes: (y_min, x_min, y_max, x_max).
    """
    h, w = image_shape
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"Tile stride must be positive. Got tile_size={tile_size}, overlap={overlap}")

    tiles = []
    y_starts = list(range(0, h, stride))
    x_starts = list(range(0, w, stride))

    # Adjust last starts so tiles don't exceed boundaries, ensuring complete coverage
    for i, y in enumerate(y_starts):
        if y + tile_size > h:
            y_starts[i] = max(0, h - tile_size)
    for j, x in enumerate(x_starts):
        if x + tile_size > w:
            x_starts[j] = max(0, w - tile_size)

    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    for y in y_starts:
        for x in x_starts:
            tiles.append((y, x, y + tile_size, x + tile_size))

    return tiles


def reconstruct_full_image_logits(
    tile_logits: List[np.ndarray],
    tile_coords: List[Tuple[int, int, int, int]],
    full_shape: Tuple[int, int],
    num_classes: int = 2,
) -> np.ndarray:
    """Stitch tiled prediction logits back into original full-image coordinate space.
    
    When overlapping tiles are used, logits are averaged across overlaps.
    For non-overlapping tiles, tiles fit into their exact coordinate slots.
    
    Args:
        tile_logits: List of logit arrays, each of shape (num_classes, tile_h, tile_w).
        tile_coords: List of (y_min, x_min, y_max, x_max) tuples.
        full_shape: (H, W) of the original image (e.g. 1024, 1024).
        num_classes: Number of prediction classes (2).
        
    Returns:
        Reconstructed full image logits of shape (num_classes, H, W).
    """
    h, w = full_shape
    accum_logits = np.zeros((num_classes, h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for logits, (ymin, xmin, ymax, xmax) in zip(tile_logits, tile_coords):
        tile_h, tile_w = ymax - ymin, xmax - xmin
        accum_logits[:, ymin:ymax, xmin:xmax] += logits[:, :tile_h, :tile_w]
        counts[ymin:ymax, xmin:xmax] += 1.0

    counts = np.maximum(counts, 1.0)
    reconstructed = accum_logits / counts[np.newaxis, :, :]
    return reconstructed
