import os
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def format_file_size(size_in_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"

def percentile_stretch(channel: np.ndarray, lower_p=2.0, upper_p=98.0) -> np.ndarray:
    """Stretches uint16 or high dynamic range rasters to clean 0-255 uint8 visual preview."""
    valid_vals = channel[channel > 0]
    if len(valid_vals) == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    
    p_low = np.percentile(valid_vals, lower_p)
    p_high = np.percentile(valid_vals, upper_p)
    if p_high <= p_low:
        p_high = p_low + 1.0
        
    stretched = np.clip((channel - p_low) / (p_high - p_low) * 255.0, 0, 255)
    return stretched.astype(np.uint8)

def read_raster_file(file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads a raster file (GeoTIFF, TIFF, PNG, JPEG) and extracts imagery data and metadata.
    Returns:
        data: (C, H, W) numpy array
        meta: dictionary with spatial & radiometric metadata
    """
    file_path = Path(file_path)
    metadata = {
        "filename": file_path.name,
        "format": file_path.suffix.lstrip(".").upper(),
        "file_size_bytes": file_path.stat().st_size,
        "file_size_formatted": format_file_size(file_path.stat().st_size),
        "crs": None,
        "bounds": None,
        "resolution": None,
        "profile": None
    }
    
    if file_path.suffix.lower() == ".npz":
        npz_data = np.load(file_path)
        img_arr = npz_data["image"]
        if img_arr.ndim == 2:
            data = img_arr[np.newaxis, :, :]
        elif img_arr.ndim == 3:
            # Check if (C, H, W) or (H, W, C)
            if img_arr.shape[0] in [1, 3, 4] and img_arr.shape[1] > 10:
                data = img_arr
            else:
                data = np.transpose(img_arr, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported npz dimensions: {img_arr.shape}")
            
        c, h, w = data.shape
        metadata["channels"] = c
        metadata["height"] = h
        metadata["width"] = w
        metadata["dtype"] = str(data.dtype)
        if "mask" in npz_data:
            metadata["has_ground_truth"] = True
        return data, metadata

    if HAS_RASTERIO and file_path.suffix.lower() in [".tif", ".tiff", ".geotiff"]:
        try:
            with rasterio.open(file_path) as src:
                data = src.read()  # (C, H, W)
                c, h, w = data.shape
                metadata["channels"] = c
                metadata["height"] = h
                metadata["width"] = w
                metadata["dtype"] = str(data.dtype)
                metadata["profile"] = src.profile.copy()
                
                if src.crs:
                    metadata["crs"] = str(src.crs)
                if src.bounds:
                    metadata["bounds"] = [
                        float(src.bounds.left),
                        float(src.bounds.bottom),
                        float(src.bounds.right),
                        float(src.bounds.top)
                    ]
                if src.res:
                    metadata["resolution"] = [float(src.res[0]), float(src.res[1])]
            return data, metadata
        except Exception:
            pass
            
    # Fallback to PIL
    with Image.open(file_path) as img:
        img_arr = np.array(img)
        if img_arr.ndim == 2:
            data = img_arr[np.newaxis, :, :]  # (1, H, W)
        elif img_arr.ndim == 3:
            data = np.transpose(img_arr, (2, 0, 1))  # (C, H, W)
        else:
            raise ValueError(f"Unsupported image dimensions: {img_arr.shape}")
            
        c, h, w = data.shape
        metadata["channels"] = c
        metadata["height"] = h
        metadata["width"] = w
        metadata["dtype"] = str(data.dtype)
        return data, metadata

def generate_preview_image(data: np.ndarray, output_path: Path) -> Path:
    """
    Generates a high-quality RGB visualization preview for the web UI.
    Handles 4-band PlanetScope (R, G, B, NIR) or 3-band RGB or 1-band grayscale.
    """
    c, h, w = data.shape
    if c >= 3:
        # Channels 0, 1, 2 correspond to R, G, B
        r = percentile_stretch(data[0])
        g = percentile_stretch(data[1])
        b = percentile_stretch(data[2])
        rgb = np.stack([r, g, b], axis=-1)
    elif c == 1:
        gray = percentile_stretch(data[0])
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        # 2 channels fallback
        ch0 = percentile_stretch(data[0])
        ch1 = percentile_stretch(data[1])
        rgb = np.stack([ch0, ch1, ch0], axis=-1)
        
    img = Image.fromarray(rgb)
    # Ensure directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return output_path

def generate_flood_overlay(original_rgb_path: Path, mask: np.ndarray, output_path: Path, alpha: float = 0.60) -> Path:
    """
    Generates a composite overlay with detected flood pixels highlighted in vivid electric cyan/blue.
    mask: (H, W) binary array (0: background, 1: flood)
    """
    with Image.open(original_rgb_path) as base_img:
        base_rgb = np.array(base_img.convert("RGB")).astype(np.float32)
        
    h, w = mask.shape
    if base_rgb.shape[:2] != (h, w):
        base_img_resized = Image.fromarray(base_rgb.astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        base_rgb = np.array(base_img_resized).astype(np.float32)
        
    # Flood highlight color: Vibrant Electric Cyan / Azure (R=0, G=220, B=255)
    flood_color = np.array([0.0, 220.0, 255.0], dtype=np.float32)
    
    overlay = base_rgb.copy()
    flood_idx = (mask == 1)
    
    overlay[flood_idx] = (1.0 - alpha) * base_rgb[flood_idx] + alpha * flood_color
    
    result_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path, format="PNG")
    return output_path

def save_geotiff_mask(mask: np.ndarray, output_path: Path, source_profile: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """
    Saves binary segmentation mask as GeoTIFF preserving source spatial reference.
    """
    if not HAS_RASTERIO:
        return None
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = mask.shape
    
    if source_profile:
        profile = source_profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            width=w,
            height=h
        )
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
        return output_path
    else:
        # Default GeoTIFF without spatial reference
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=1,
            dtype=rasterio.uint8
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)
        return output_path
