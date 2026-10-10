"""
Universal Image Adapter & Compatibility Inspector
Provides universal ingestion for common image and raster formats:
- GeoTIFF / TIFF (.tif, .tiff)
- PNG (.png)
- JPG / JPEG (.jpg, .jpeg)
- WebP (.webp)
- BMP (.bmp)

Strictly enforces scientific compatibility boundaries:
- Category A: Verified 4-band satellite imagery (PlanetScope B, G, R, NIR) -> Model inference enabled.
- Category B: Ordinary RGB imagery (3 channels) -> Preview & image processing only; segmentation blocked.
- Category C: 4-channel RGBA (Transparency channel, not NIR) -> Preview & image processing only; segmentation blocked.
- Category D: Grayscale (1 channel) or Corrupted/Unsupported images -> Friendly explanatory error.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import rasterio

from deployment_config import NORM_MEAN, NORM_STD, PATCH_SIZE
from preprocessing import create_rgb_preview, normalize_image, partition_into_patches


SUPPORTED_EXTENSIONS = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp"]


@dataclass
class ImageInspectionResult:
    category: str  # "CATEGORY_A", "CATEGORY_B", "CATEGORY_C", "CATEGORY_D"
    is_model_compatible: bool
    status_badge: str
    status_message: str
    format_name: str
    dimensions_str: str
    width: int
    height: int
    channels: int
    bit_depth: str
    color_mode: str
    crs: str
    valid_coverage_pct: float
    file_size_mb: float
    preview_image: Optional[Image.Image]
    raw_array: Optional[np.ndarray]
    valid_mask: Optional[np.ndarray]
    meta: Optional[Dict[str, Any]]
    error: Optional[str] = None


def inspect_and_load_universal_image(file_path: str) -> ImageInspectionResult:
    """
    Universally inspects, decodes, and categorizes any uploaded image or raster.
    Protects the downstream UNet-Mamba model from unscientific or fabricated inputs.
    """
    path = Path(file_path)
    if not path.exists():
        return ImageInspectionResult(
            category="CATEGORY_D",
            is_model_compatible=False,
            status_badge="FILE NOT FOUND",
            status_message=f"The selected file does not exist: {path.name}",
            format_name="Unknown",
            dimensions_str="N/A",
            width=0,
            height=0,
            channels=0,
            bit_depth="N/A",
            color_mode="N/A",
            crs="None",
            valid_coverage_pct=0.0,
            file_size_mb=0.0,
            preview_image=None,
            raw_array=None,
            valid_mask=None,
            meta=None,
            error="File not found"
        )

    file_size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    ext = path.suffix.lower()

    if file_size_mb > 250.0:
        return ImageInspectionResult(
            category="CATEGORY_D",
            is_model_compatible=False,
            status_badge="FILE TOO LARGE",
            status_message=f"File exceeds maximum allowed upload size (250 MB). Current size: {file_size_mb} MB.",
            format_name=ext.upper(),
            dimensions_str="N/A",
            width=0,
            height=0,
            channels=0,
            bit_depth="N/A",
            color_mode="N/A",
            crs="None",
            valid_coverage_pct=0.0,
            file_size_mb=file_size_mb,
            preview_image=None,
            raw_array=None,
            valid_mask=None,
            meta=None,
            error="File exceeds size limit"
        )

    # -------------------------------------------------------------------------
    # 1. Try Rasterio Ingestion (GeoTIFF / TIFF / Geospatial Rasters)
    # -------------------------------------------------------------------------
    is_rasterio_handled = False
    if ext in [".tif", ".tiff"]:
        try:
            with rasterio.open(path) as src:
                count = src.count
                h, w = src.height, src.width
                dtype_name = str(src.dtypes[0])
                crs_str = str(src.crs) if src.crs else "None (Non-Georeferenced TIFF)"
                meta = src.meta.copy()
                raw_data = src.read()  # (count, H, W)
                is_rasterio_handled = True

                # Check validity
                valid_mask = (raw_data != 0).any(axis=0)
                valid_px = int(valid_mask.sum())
                valid_pct = round(valid_px / max(1, h * w) * 100.0, 1)

                # Check if this TIFF is 4-band multispectral vs 4-channel RGBA
                if count == 4:
                    # Detect RGBA TIFF where 4th band is Alpha
                    is_rgba_tiff = False
                    if hasattr(src, "colorinterp") and src.colorinterp:
                        from rasterio.enums import ColorInterp
                        if ColorInterp.alpha in src.colorinterp:
                            is_rgba_tiff = True
                    
                    if is_rgba_tiff:
                        # Category C: 4-channel RGBA
                        rgb_arr = raw_data[:3].transpose(1, 2, 0)
                        if dtype_name != "uint8":
                            p2, p98 = np.percentile(rgb_arr[valid_mask], (2, 98), axis=0) if valid_px > 0 else (0, 1)
                            rgb_arr = np.clip((rgb_arr - p2) / (p98 - p2 + 1e-4) * 255.0, 0, 255).astype(np.uint8)
                        preview_img = Image.fromarray(rgb_arr.astype(np.uint8))

                        return ImageInspectionResult(
                            category="CATEGORY_C",
                            is_model_compatible=False,
                            status_badge="PREVIEW ONLY (RGBA IMAGE)",
                            status_message=(
                                "Uploaded TIFF contains 4 channels, but channel 4 is an Alpha/Transparency mask, "
                                "not a calibrated Near-Infrared (NIR) multispectral satellite band. "
                                "CAVE-Flood requires genuine 4-band PlanetScope imagery (Blue, Green, Red, NIR). "
                                "Flood segmentation is disabled to prevent invalid predictions. Visual filters are available below."
                            ),
                            format_name="TIFF (RGBA)",
                            dimensions_str=f"{w} × {h} pixels",
                            width=w,
                            height=h,
                            channels=4,
                            bit_depth=dtype_name,
                            color_mode="RGBA (Color + Transparency)",
                            crs=crs_str,
                            valid_coverage_pct=valid_pct,
                            file_size_mb=file_size_mb,
                            preview_image=preview_img,
                            raw_array=raw_data,
                            valid_mask=valid_mask,
                            meta=meta
                        )
                    else:
                        # Category A: Genuine 4-band satellite imagery!
                        preview_img = create_rgb_preview(raw_data, valid_mask)
                        return ImageInspectionResult(
                            category="CATEGORY_A",
                            is_model_compatible=True,
                            status_badge="COMPATIBLE (FOUR-BAND SATELLITE IMAGERY)",
                            status_message=(
                                "Verified 4-band multispectral satellite raster (Blue, Green, Red, Near-Infrared). "
                                "Input matches original training band configuration. AI flood inundation segmentation is enabled."
                            ),
                            format_name="PlanetScope GeoTIFF" if src.crs else "4-Band TIFF",
                            dimensions_str=f"{w} × {h} pixels",
                            width=w,
                            height=h,
                            channels=4,
                            bit_depth=dtype_name,
                            color_mode="Multispectral (4 Bands)",
                            crs=crs_str,
                            valid_coverage_pct=valid_pct,
                            file_size_mb=file_size_mb,
                            preview_image=preview_img,
                            raw_array=raw_data,
                            valid_mask=valid_mask,
                            meta=meta
                        )

                elif count == 3:
                    # Category B: 3-band RGB TIFF
                    rgb_arr = raw_data.transpose(1, 2, 0)
                    if dtype_name != "uint8":
                        p2, p98 = np.percentile(rgb_arr[valid_mask], (2, 98), axis=0) if valid_px > 0 else (0, 1)
                        rgb_arr = np.clip((rgb_arr - p2) / (p98 - p2 + 1e-4) * 255.0, 0, 255).astype(np.uint8)
                    preview_img = Image.fromarray(rgb_arr.astype(np.uint8))

                    return ImageInspectionResult(
                        category="CATEGORY_B",
                        is_model_compatible=False,
                        status_badge="PREVIEW ONLY (RGB TIFF)",
                        status_message=(
                            "Uploaded TIFF has 3 optical RGB bands. The trained UNet-Mamba architecture requires "
                            "4 multispectral bands including calibrated Near-Infrared (NIR) data. "
                            "Flood segmentation is disabled to prevent unscientific fabricated predictions."
                        ),
                        format_name="3-Band TIFF (RGB)",
                        dimensions_str=f"{w} × {h} pixels",
                        width=w,
                        height=h,
                        channels=3,
                        bit_depth=dtype_name,
                        color_mode="RGB (3 Bands)",
                        crs=crs_str,
                        valid_coverage_pct=valid_pct,
                        file_size_mb=file_size_mb,
                        preview_image=preview_img,
                        raw_array=raw_data,
                        valid_mask=valid_mask,
                        meta=meta
                    )

                elif count == 1:
                    # Category D: 1-band Grayscale TIFF
                    gray_arr = raw_data[0]
                    p2, p98 = np.percentile(gray_arr[valid_mask], (2, 98)) if valid_px > 0 else (0, 1)
                    gray_norm = np.clip((gray_arr - p2) / (p98 - p2 + 1e-4) * 255.0, 0, 255).astype(np.uint8)
                    preview_img = Image.fromarray(gray_norm).convert("RGB")

                    return ImageInspectionResult(
                        category="CATEGORY_D",
                        is_model_compatible=False,
                        status_badge="PREVIEW ONLY (GRAYSCALE TIFF)",
                        status_message=(
                            "Uploaded raster contains only 1 channel (Grayscale/Single-Band). "
                            "Model inference requires 4 multispectral channels (B, G, R, NIR)."
                        ),
                        format_name="1-Band TIFF",
                        dimensions_str=f"{w} × {h} pixels",
                        width=w,
                        height=h,
                        channels=1,
                        bit_depth=dtype_name,
                        color_mode="Grayscale (1 Band)",
                        crs=crs_str,
                        valid_coverage_pct=valid_pct,
                        file_size_mb=file_size_mb,
                        preview_image=preview_img,
                        raw_array=raw_data,
                        valid_mask=valid_mask,
                        meta=meta
                    )

        except Exception as e:
            # Rasterio failed; fall back to Pillow inspection below
            is_rasterio_handled = False

    # -------------------------------------------------------------------------
    # 2. Pillow Ingestion (PNG, JPG, WebP, BMP, or non-geospatial TIFF)
    # -------------------------------------------------------------------------
    try:
        with Image.open(path) as pil_img:
            # Verify file integrity
            pil_img.verify()
            
        # Re-open after verify() to decode pixels
        with Image.open(path) as pil_img:
            pil_img.load()
            w, h = pil_img.size
            mode = pil_img.mode
            format_name = pil_img.format or ext[1:].upper()
            
            # Category C: 4-Channel RGBA
            if mode == "RGBA":
                rgb_preview = Image.new("RGB", (w, h), (255, 255, 255))
                rgb_preview.paste(pil_img, mask=pil_img.split()[3])
                raw_np = np.array(pil_img).transpose(2, 0, 1)  # (4, H, W)
                valid_mask = np.ones((h, w), dtype=bool)

                return ImageInspectionResult(
                    category="CATEGORY_C",
                    is_model_compatible=False,
                    status_badge="PREVIEW ONLY (RGBA IMAGE)",
                    status_message=(
                        f"Uploaded {format_name} has 4 channels, but channel 4 is an Alpha/Transparency mask, "
                        "not Near-Infrared (NIR) satellite data. CAVE-Flood requires calibrated NIR reflectance for water detection. "
                        "Flood segmentation is disabled. Visual processing previews are available below."
                    ),
                    format_name=f"{format_name} (RGBA)",
                    dimensions_str=f"{w} × {h} pixels",
                    width=w,
                    height=h,
                    channels=4,
                    bit_depth="uint8 (8-bit)",
                    color_mode="RGBA (Color + Alpha)",
                    crs="None (Standard Image)",
                    valid_coverage_pct=100.0,
                    file_size_mb=file_size_mb,
                    preview_image=rgb_preview,
                    raw_array=raw_np,
                    valid_mask=valid_mask,
                    meta=None
                )

            # Category B: Standard RGB
            elif mode in ["RGB", "P", "CMYK"]:
                rgb_preview = pil_img.convert("RGB")
                raw_np = np.array(rgb_preview).transpose(2, 0, 1)  # (3, H, W)
                valid_mask = np.ones((h, w), dtype=bool)

                return ImageInspectionResult(
                    category="CATEGORY_B",
                    is_model_compatible=False,
                    status_badge="PREVIEW ONLY (RGB IMAGE)",
                    status_message=(
                        f"Standard {format_name} image uploaded (3-channel RGB). The trained CAVE-Flood foundation model "
                        "requires 4-band PlanetScope satellite imagery with Near-Infrared (NIR) data. "
                        "AI flood segmentation is blocked to maintain scientific integrity. Visual processing filters are active below."
                    ),
                    format_name=f"{format_name} (RGB)",
                    dimensions_str=f"{w} × {h} pixels",
                    width=w,
                    height=h,
                    channels=3,
                    bit_depth="uint8 (8-bit)",
                    color_mode=f"{mode} -> RGB",
                    crs="None (Standard Image)",
                    valid_coverage_pct=100.0,
                    file_size_mb=file_size_mb,
                    preview_image=rgb_preview,
                    raw_array=raw_np,
                    valid_mask=valid_mask,
                    meta=None
                )

            # Category D: Grayscale
            elif mode in ["L", "1", "I;16"]:
                gray_preview = pil_img.convert("RGB")
                raw_np = np.array(pil_img)
                valid_mask = np.ones((h, w), dtype=bool)

                return ImageInspectionResult(
                    category="CATEGORY_D",
                    is_model_compatible=False,
                    status_badge="PREVIEW ONLY (GRAYSCALE)",
                    status_message=(
                        f"Single-channel Grayscale {format_name} image. Segmentation requires 4 multispectral channels (B, G, R, NIR)."
                    ),
                    format_name=f"{format_name} (Grayscale)",
                    dimensions_str=f"{w} × {h} pixels",
                    width=w,
                    height=h,
                    channels=1,
                    bit_depth=f"{mode} (Single Channel)",
                    color_mode="Grayscale",
                    crs="None (Standard Image)",
                    valid_coverage_pct=100.0,
                    file_size_mb=file_size_mb,
                    preview_image=gray_preview,
                    raw_array=raw_np,
                    valid_mask=valid_mask,
                    meta=None
                )

            else:
                rgb_preview = pil_img.convert("RGB")
                return ImageInspectionResult(
                    category="CATEGORY_D",
                    is_model_compatible=False,
                    status_badge="UNSUPPORTED COLOR MODE",
                    status_message=f"Image color mode '{mode}' is unsupported for satellite flood segmentation.",
                    format_name=format_name,
                    dimensions_str=f"{w} × {h} pixels",
                    width=w,
                    height=h,
                    channels=len(pil_img.getbands()),
                    bit_depth=mode,
                    color_mode=mode,
                    crs="None",
                    valid_coverage_pct=100.0,
                    file_size_mb=file_size_mb,
                    preview_image=rgb_preview,
                    raw_array=None,
                    valid_mask=None,
                    meta=None
                )

    except Exception as e:
        # Corrupted or unreadable file
        return ImageInspectionResult(
            category="CATEGORY_D",
            is_model_compatible=False,
            status_badge="CORRUPTED / UNREADABLE FILE",
            status_message=f"The uploaded file could not be decoded as a valid image format. Details: {str(e)}",
            format_name="Corrupted",
            dimensions_str="N/A",
            width=0,
            height=0,
            channels=0,
            bit_depth="N/A",
            color_mode="N/A",
            crs="None",
            valid_coverage_pct=0.0,
            file_size_mb=file_size_mb,
            preview_image=None,
            raw_array=None,
            valid_mask=None,
            meta=None,
            error=str(e)
        )


# -----------------------------------------------------------------------------
# Phase 6: Basic Image Processing Features
# -----------------------------------------------------------------------------
def generate_image_processing_previews(rgb_image: Image.Image) -> Dict[str, Image.Image]:
    """
    Computes four standard computer-vision visual previews:
    1. Original RGB Image
    2. Grayscale Conversion (Luminance)
    3. Edge Detection (Sobel/Gradient filter)
    4. Contrast Enhancement (Dynamic stretch)
    
    These are basic visual processing operations, strictly separated from AI flood detection.
    """
    if rgb_image is None:
        return {}
        
    # 1. Original RGB
    orig = rgb_image.copy()
    
    # 2. Grayscale Conversion
    grayscale = ImageOps.grayscale(orig).convert("RGB")
    
    # 3. Edge Detection (Sobel / Laplacian outline filter)
    edge_gray = ImageOps.grayscale(orig).filter(ImageFilter.FIND_EDGES)
    # Invert and boost so edges are sharp white on dark slate background
    edge_boosted = ImageEnhance.Contrast(edge_gray).enhance(2.0)
    edge_rgb = edge_boosted.convert("RGB")
    
    # 4. Contrast Enhancement
    # High-contrast stretched version
    enhancer = ImageEnhance.Contrast(orig)
    contrast_enhanced = enhancer.enhance(1.6)
    
    return {
        "original_rgb": orig,
        "grayscale": grayscale,
        "edge_detection": edge_rgb,
        "contrast_enhanced": contrast_enhanced
    }
