"""
Visualization & Export Module
Generates high-contrast flood segmentation masks, semi-transparent satellite overlays,
multi-model comparative panels, XAI diagnostic dashboards, and geospatial GeoTIFF exports.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio


def create_segmentation_mask_image(pred_mask, valid_mask):
    """
    Renders high-contrast binary flood segmentation map.
    Palette:
        - Flood Water (1): Ocean Blue [0, 162, 255]
        - Land / Non-flood (0): Dark Slate [30, 41, 59]
        - Invalid / Nodata: Pure Black [0, 0, 0]
    """
    h, w = pred_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Non-flood background
    rgb[(pred_mask == 0) & valid_mask] = [30, 41, 59]
    # Inundated water body
    rgb[(pred_mask == 1) & valid_mask] = [0, 162, 255]
    
    return Image.fromarray(rgb)


def create_flood_overlay(rgb_preview, pred_mask, valid_mask, alpha=0.45):
    """
    Blends semi-transparent flood overlay onto the satellite optical preview,
    preserving terrain textures and structures beneath water layers.
    """
    rgb_arr = np.array(rgb_preview).astype(np.float32)
    h, w, _ = rgb_arr.shape
    
    flood_color = np.array([0.0, 162.0, 255.0], dtype=np.float32)
    flood_pixels = (pred_mask == 1) & valid_mask
    
    blended = rgb_arr.copy()
    blended[flood_pixels] = (1.0 - alpha) * rgb_arr[flood_pixels] + alpha * flood_color
    
    # Boundary enhancement (sharpen outer edge of water)
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(flood_pixels)
    edges = dilated & (~flood_pixels) & valid_mask
    blended[edges] = 0.5 * rgb_arr[edges] + 0.5 * np.array([255.0, 255.0, 255.0])
    
    blended[~valid_mask] = 0.0
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8))


def create_comparison_panel(rgb_preview, pred_base, pred_refined, valid_mask, base_title="E0 UNet-Mamba", ref_title="E2 RemoteCLIP CASR"):
    """
    Generates a 4-panel comparison:
    1. Satellite RGB Preview
    2. Baseline Prediction
    3. Refined Prediction
    4. Difference Transitions (Cyan: Water Added, Magenta: False Alarms Removed)
    """
    h, w = pred_base.shape
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#ffffff")
    plt.subplots_adjust(wspace=0.08, hspace=0.0)
    
    # Panel 1: RGB
    axes[0].imshow(rgb_preview)
    axes[0].set_title("PlanetScope RGB Imagery", fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    axes[0].axis("off")
    
    # Panel 2: Baseline
    base_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    base_rgb[(pred_base == 0) & valid_mask] = [30, 41, 59]
    base_rgb[(pred_base == 1) & valid_mask] = [0, 162, 255]
    axes[1].imshow(base_rgb)
    base_pct = ((pred_base == 1) & valid_mask).sum() / max(1, valid_mask.sum()) * 100.0
    axes[1].set_title(f"{base_title} ({base_pct:.1f}%)", fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    axes[1].axis("off")
    
    # Panel 3: Refined
    ref_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    ref_rgb[(pred_refined == 0) & valid_mask] = [30, 41, 59]
    ref_rgb[(pred_refined == 1) & valid_mask] = [0, 162, 255]
    axes[2].imshow(ref_rgb)
    ref_pct = ((pred_refined == 1) & valid_mask).sum() / max(1, valid_mask.sum()) * 100.0
    axes[2].set_title(f"{ref_title} ({ref_pct:.1f}%)", fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    axes[2].axis("off")
    
    # Panel 4: CASR Transition Difference
    diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    diff_rgb[(pred_base == pred_refined) & valid_mask] = [40, 50, 70]
    # Added flood: Cyan [0, 220, 255]
    diff_rgb[(pred_base == 0) & (pred_refined == 1) & valid_mask] = [0, 220, 255]
    # Suppressed false alarms: Magenta [255, 60, 180]
    diff_rgb[(pred_base == 1) & (pred_refined == 0) & valid_mask] = [255, 60, 180]
    
    axes[3].imshow(diff_rgb)
    added_n = int(((pred_base == 0) & (pred_refined == 1) & valid_mask).sum())
    supp_n = int(((pred_base == 1) & (pred_refined == 0) & valid_mask).sum())
    axes[3].set_title(f"CASR Edits (+{added_n:,} / -{supp_n:,})", fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    axes[3].axis("off")
    
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba[:, :, :3])


def create_xai_dashboard_image(entropy_map, cand_mask, gate_map, guidance_map, prob_delta, layercam_map, valid_mask):
    """
    Renders an audited 6-panel XAI diagnostic dashboard:
    1. Predictive Entropy (Baseline Uncertainty)
    2. Candidate Region Mask (Top 25% Gating)
    3. CASR Gate Activation (Dynamic Gating Strength)
    4. RemoteCLIP Guidance Margin
    5. Residual Probability Delta
    6. LayerCAM Attribution Heatmap
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#ffffff")
    plt.subplots_adjust(wspace=0.12, hspace=0.18)
    
    # 1. Entropy
    ent_disp = np.ma.masked_where(~valid_mask, entropy_map)
    im1 = axes[0, 0].imshow(ent_disp, cmap="magma", vmin=0, vmax=1)
    axes[0, 0].set_title("1. Predictive Entropy Map (Uncertainty)", fontsize=11, fontweight="bold", color="#0f172a")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 2. Candidate Mask
    cand_disp = np.ma.masked_where(~valid_mask, cand_mask)
    im2 = axes[0, 1].imshow(cand_disp, cmap="Blues", vmin=0, vmax=1)
    axes[0, 1].set_title("2. Top 25% Candidate Boundary Mask", fontsize=11, fontweight="bold", color="#0f172a")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Gate Map
    gate_disp = np.ma.masked_where(~valid_mask, gate_map)
    im3 = axes[0, 2].imshow(gate_disp, cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].set_title("3. CASR Dynamic Gate Activation", fontsize=11, fontweight="bold", color="#0f172a")
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. RemoteCLIP Guidance
    guid_disp = np.ma.masked_where(~valid_mask, guidance_map)
    im4 = axes[1, 0].imshow(guid_disp, cmap="coolwarm")
    axes[1, 0].set_title("4. RemoteCLIP Semantic Guidance Margin", fontsize=11, fontweight="bold", color="#0f172a")
    axes[1, 0].axis("off")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 5. Residual Delta
    delta_disp = np.ma.masked_where(~valid_mask, prob_delta)
    vmax = max(0.01, float(np.abs(delta_disp).max()))
    im5 = axes[1, 1].imshow(delta_disp, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title("5. Residual Probability Delta (ΔP)", fontsize=11, fontweight="bold", color="#0f172a")
    axes[1, 1].axis("off")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 6. LayerCAM
    if layercam_map is not None:
        cam_disp = np.ma.masked_where(~valid_mask, layercam_map)
        im6 = axes[1, 2].imshow(cam_disp, cmap="plasma", vmin=0, vmax=1)
        axes[1, 2].set_title("6. LayerCAM Attribution (CASR block2)", fontsize=11, fontweight="bold", color="#0f172a")
    else:
        axes[1, 2].text(0.5, 0.5, "LayerCAM Not Requested", ha="center", va="center", color="#64748b", fontsize=12)
        im6 = None
        
    axes[1, 2].axis("off")
    if im6 is not None:
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba[:, :, :3])


def save_geotiff_output(pred_mask, meta, output_path):
    """
    Exports predicted segmentation map into GeoTIFF format preserving original
    Coordinate Reference System (CRS) and geospatial affine transform.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_meta = meta.copy()
    out_meta.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": 255
    })
    
    export_mask = pred_mask.astype(np.uint8)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(export_mask, 1)
        
    return str(out_path)
