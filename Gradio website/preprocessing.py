"""
Satellite Imagery Preprocessing & Validation Module
Handles 4-band PlanetScope GeoTIFF ingestion, validity checks, RGB contrast stretching,
channel normalization, and spatial tiling/reconstruction.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import rasterio

from deployment_config import NORM_MEAN, NORM_STD, PATCH_SIZE


def validate_and_load_geotiff(file_path):
    """
    Validates and loads a four-band PlanetScope GeoTIFF raster.
    
    Returns:
        dict containing:
            - 'image': np.ndarray of shape (4, H, W)
            - 'valid_mask': np.ndarray of shape (H, W) (bool)
            - 'meta': rasterio metadata dict (crs, transform, width, height, etc.)
            - 'info': human-readable summary dict
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded file does not exist: {path}")
        
    try:
        with rasterio.open(path) as src:
            count = src.count
            h, w = src.height, src.width
            dtype_name = src.dtypes[0]
            crs_str = str(src.crs) if src.crs else "Unprojected / Local"
            bounds = src.bounds
            transform = src.transform
            meta = src.meta.copy()
            
            if count != 4:
                raise ValueError(
                    f"Unsupported band count: {count}. CAVE-Flood requires exactly 4-band PlanetScope "
                    f"multispectral imagery (Blue, Green, Red, Near-Infrared). Ordinary 3-band RGB or 1-band rasters "
                    f"cannot be processed by the 4-channel foundation model."
                )
                
            if h < 64 or w < 64:
                raise ValueError(f"Raster spatial dimensions ({w}x{h}) are too small for processing (min 64x64).")
                
            raw_data = src.read()  # (4, H, W)
            
    except rasterio.errors.RasterioIOError as e:
        raise ValueError(f"Failed to open GeoTIFF raster: {str(e)}")
        
    # Valid data detection (non-zero in at least one band)
    valid_mask = (raw_data != 0).any(axis=0)
    valid_pixel_count = int(valid_mask.sum())
    total_pixels = h * w
    valid_pct = (valid_pixel_count / total_pixels * 100.0) if total_pixels > 0 else 0.0
    
    if valid_pixel_count == 0:
        raise ValueError("The uploaded raster contains only zeros or nodata pixels. No valid imagery to segment.")
        
    info = {
        "dimensions": f"{w} × {h} pixels",
        "bands": f"{count} Multispectral Channels (B, G, R, NIR)",
        "bit_depth": dtype_name,
        "crs": crs_str,
        "valid_coverage": f"{valid_pct:.1f}% ({valid_pixel_count:,} valid pixels)",
        "file_size_mb": f"{path.stat().st_size / (1024 * 1024):.2f} MB"
    }
    
    return {
        "image": raw_data,
        "valid_mask": valid_mask,
        "meta": meta,
        "transform": transform,
        "info": info
    }


def create_rgb_preview(image_array, valid_mask=None):
    """
    Renders high-fidelity true-color RGB preview from 4-band PlanetScope raster.
    PlanetScope band order: Band 1=Blue, Band 2=Green, Band 3=Red, Band 4=NIR.
    True-color RGB extraction uses bands [Red=2, Green=1, Blue=0].
    Applies 2%-98% percentile contrast stretch.
    """
    c, h, w = image_array.shape
    if c >= 3:
        # Standard PlanetScope optical RGB [Red, Green, Blue] = channels [2, 1, 0]
        rgb = image_array[[2, 1, 0], :, :].transpose(1, 2, 0).astype(np.float32)
    else:
        rgb = np.repeat(image_array[0, :, :, np.newaxis], 3, axis=2).astype(np.float32)
        
    if valid_mask is None:
        valid_mask = (image_array != 0).any(axis=0)
        
    if valid_mask.sum() > 0:
        p2, p98 = np.percentile(rgb[valid_mask], (2, 98), axis=0)
    else:
        p2, p98 = rgb.min(axis=(0, 1)), rgb.max(axis=(0, 1))
        
    p98 = np.maximum(p98, p2 + 1e-4)
    rgb_stretched = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    rgb_stretched[~valid_mask] = 0.0
    
    rgb_uint8 = (rgb_stretched * 255.0).astype(np.uint8)
    return Image.fromarray(rgb_uint8)


def normalize_image(image_array):
    """
    Applies training-fitted channel normalization to 4-band PlanetScope imagery.
    """
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(4, 1, 1)
    std = np.array(NORM_STD, dtype=np.float32).reshape(4, 1, 1)
    return (image_array.astype(np.float32) - mean) / (std + 1e-6)


def partition_into_patches(image_array, valid_mask=None, patch_size=PATCH_SIZE):
    """
    Deconstructs image into a grid of 512x512 patches with exact spatial coordinates.
    Handles arbitrary resolutions with boundary padding and coordinate offsets.
    """
    c, h, w = image_array.shape
    
    # Calculate coordinate steps
    y_starts = list(range(0, max(1, h - patch_size + 1), patch_size))
    if y_starts[-1] + patch_size < h:
        y_starts.append(h - patch_size)
    if h < patch_size:
        y_starts = [0]
        
    x_starts = list(range(0, max(1, w - patch_size + 1), patch_size))
    if x_starts[-1] + patch_size < w:
        x_starts.append(w - patch_size)
    if w < patch_size:
        x_starts = [0]
        
    patches = []
    for y in y_starts:
        for x in x_starts:
            # Crop or pad patch
            patch_data = np.zeros((c, patch_size, patch_size), dtype=image_array.dtype)
            h_slice = min(patch_size, h - y)
            w_slice = min(patch_size, w - x)
            
            patch_data[:, :h_slice, :w_slice] = image_array[:, y:y+h_slice, x:x+w_slice]
            
            if valid_mask is not None:
                p_valid = np.zeros((patch_size, patch_size), dtype=bool)
                p_valid[:h_slice, :w_slice] = valid_mask[y:y+h_slice, x:x+w_slice]
            else:
                p_valid = (patch_data != 0).any(axis=0)
                
            patches.append({
                "top": y,
                "left": x,
                "h_actual": h_slice,
                "w_actual": w_slice,
                "image": patch_data,
                "valid_mask": p_valid
            })
            
    return patches, (h, w)


def validate_and_load_groundtruth(label_path, expected_shape):
    """
    Validates and maps an optional ground-truth mask.
    Map rules: Raw 1 -> 0 (Non-flood), Raw 2 -> 1 (Flood), others -> -1 (Ignore).
    """
    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file does not exist: {path}")
        
    with rasterio.open(path) as src:
        raw_mask = src.read(1)
        
    exp_h, exp_w = expected_shape
    if raw_mask.shape != (exp_h, exp_w):
        raise ValueError(
            f"Ground-truth dimensions ({raw_mask.shape[1]}x{raw_mask.shape[0]}) do not match "
            f"the satellite imagery dimensions ({exp_w}x{exp_h})."
        )
        
    mapped_mask = np.full_like(raw_mask, fill_value=-1, dtype=np.int64)
    # Binary FloodPlanet standard encoding:
    mapped_mask[raw_mask == 1] = 0   # Background / Land
    mapped_mask[raw_mask == 2] = 1   # Flood Water
    
    # Also support 0/1 binary masks if user uploaded direct 0/1 mask
    if (raw_mask == 2).sum() == 0 and (raw_mask == 1).sum() > 0 and (raw_mask == 0).sum() > 0:
        mapped_mask[raw_mask == 0] = 0
        mapped_mask[raw_mask == 1] = 1
        
    return mapped_mask
