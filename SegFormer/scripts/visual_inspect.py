#!/usr/bin/env python3
"""Visual Inspection CLI Tool for FloodPlanet Dataset.

Generates comprehensive 7-panel diagnostic figures showing:
1. Original Satellite Image (Sentinel-1 SAR / Sentinel-2 Optical)
2. Modified Satellite Image (PlanetScope 3m True Color RGB)
3. Original Ground-Truth Mask (Raw {1, 2})
4. Processed Ground-Truth Mask (Target {0, 1})
5. Image-Mask Overlay (RGB + Flood Highlight)
6. Original vs Modified Comparison (SAR/Optical alignment)
7. Raw vs Processed Mask Comparison (Value preservation verification)

Also generates and updates a browsable HTML gallery at:
outputs/visual_inspection/index.html

Supports --open to automatically launch Windows Photo Viewer or default browser.
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tifffile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.label_processing import map_labels
from src.data.preprocessing import extract_rgb

DATASET_ROOT = os.path.join(PROJECT_ROOT, "Datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "visual_inspection")


def render_7panel(
    event: str,
    chip_name: str,
    output_png: str,
) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    ps_path = os.path.join(DATASET_ROOT, event, "PS", chip_name)
    lbl_path = os.path.join(DATASET_ROOT, event, "labels", chip_name)

    if not (os.path.exists(ps_path) and os.path.exists(lbl_path)):
        print(f"Error: Missing files for {event}/{chip_name}")
        return False

    ps_raw = tifffile.imread(ps_path)
    lbl_raw = tifffile.imread(lbl_path)

    # 1. Modified Optical RGB (PlanetScope Bands [2, 1, 0])
    ps_rgb = extract_rgb(ps_raw, rgb_bands=(2, 1, 0))
    disp_ps_rgb = np.clip(ps_rgb.astype(np.float32) / 255.0, 0.0, 1.0)

    # 2. Original Satellite Image (S1 SAR VV/VH or S2 Optical)
    s1_path = os.path.join(DATASET_ROOT, event, "S1", chip_name)
    s2_path = os.path.join(DATASET_ROOT, event, "S2", chip_name)
    disp_orig = None
    orig_label = "Original: Not Available"

    if os.path.exists(s1_path):
        s1_raw = tifffile.imread(s1_path)  # (2, H, W) float32 in dB
        vv = s1_raw[0] if s1_raw.ndim == 3 else s1_raw
        # Scale dB [-30, 0] to [0, 1]
        disp_orig = np.clip((vv + 30.0) / 30.0, 0.0, 1.0)
        orig_label = "1. Original: S1 SAR (VV dB)"
    elif os.path.exists(s2_path):
        s2_raw = tifffile.imread(s2_path)
        if s2_raw.ndim == 3 and s2_raw.shape[0] >= 3:
            s2_rgb = np.stack([s2_raw[2], s2_raw[1], s2_raw[0]], axis=-1)
            disp_orig = np.clip(s2_rgb * 2.5, 0.0, 1.0)
            orig_label = "1. Original: S2 Optical (RGB)"

    if disp_orig is None:
        disp_orig = np.zeros((334, 334), dtype=np.float32)
        orig_label = "1. Original: None"

    # 3. Raw Ground Truth Mask
    raw_int = np.rint(lbl_raw).astype(np.int64)

    # 4. Processed Target Mask (0: Land, 1: Flood)
    proc_mask = map_labels(lbl_raw, ignore_index=255)

    # 5. Image-Mask Overlay
    overlay = disp_ps_rgb.copy()
    flood_mask = (proc_mask == 1)
    alpha = 0.45
    overlay[flood_mask] = (1 - alpha) * overlay[flood_mask] + alpha * np.array([0.1, 0.45, 0.95], dtype=np.float32)

    # 6. Original vs Modified Comparison (Side by side / composite)
    # Resize original to match PS dimensions for visual comparison
    from scipy.ndimage import zoom
    zoom_y = disp_ps_rgb.shape[0] / disp_orig.shape[0]
    zoom_x = disp_ps_rgb.shape[1] / disp_orig.shape[1]
    resized_orig = zoom(disp_orig, (zoom_y, zoom_x), order=1)
    # Blend Orig Grayscale with PS RGB
    orig_3ch = np.stack([resized_orig]*3, axis=-1)
    orig_vs_mod = 0.5 * orig_3ch + 0.5 * disp_ps_rgb

    # 7. Raw vs Processed Mask Difference Map
    # Shows conservation: Raw 1 (Land) -> Target 0, Raw 2 (Flood) -> Target 1
    # Green = Exact conservation agreement
    # Red = Mismatch (should be 0)
    raw_flood = (raw_int == 2)
    proc_flood = (proc_mask == 1)
    diff_rgb = np.zeros((raw_int.shape[0], raw_int.shape[1], 3), dtype=np.float32)
    agreement = (raw_flood == proc_flood)
    diff_rgb[agreement] = [0.1, 0.7, 0.2]   # Green = 100% Agreement
    diff_rgb[~agreement] = [0.9, 0.1, 0.1]  # Red = Discrepancy

    # Plot 7 panels
    fig, axes = plt.subplots(1, 7, figsize=(28, 4), dpi=140)

    # Panel 1: Original Satellite
    axes[0].imshow(disp_orig, cmap="gray" if disp_orig.ndim == 2 else None)
    axes[0].set_title(orig_label, fontsize=9)
    axes[0].axis("off")

    # Panel 2: Modified Satellite (PS RGB)
    axes[1].imshow(disp_ps_rgb)
    axes[1].set_title("2. Mod: PlanetScope (RGB)", fontsize=9)
    axes[1].axis("off")

    # Panel 3: Raw Mask
    axes[2].imshow(raw_int, cmap="viridis", vmin=0, vmax=2)
    axes[2].set_title("3. Raw Mask (1=Land, 2=Flood)", fontsize=9)
    axes[2].axis("off")

    # Panel 4: Processed Mask
    axes[3].imshow(proc_mask == 1, cmap="Blues", vmin=0, vmax=1)
    flood_pct = (proc_mask == 1).sum() / proc_mask.size * 100.0
    axes[3].set_title(f"4. Target Mask ({flood_pct:.1f}% Flood)", fontsize=9)
    axes[3].axis("off")

    # Panel 5: Overlay
    axes[4].imshow(overlay)
    axes[4].set_title("5. RGB + Flood Overlay", fontsize=9)
    axes[4].axis("off")

    # Panel 6: Original vs Modified
    axes[5].imshow(np.clip(orig_vs_mod, 0.0, 1.0))
    axes[5].set_title("6. Orig vs Mod Blend", fontsize=9)
    axes[5].axis("off")

    # Panel 7: Raw vs Processed Comparison
    axes[6].imshow(diff_rgb)
    axes[6].set_title("7. Raw vs Target Agreement\n(Green = Exact Match)", fontsize=9)
    axes[6].axis("off")

    plt.suptitle(f"Event: {event} | Chip: {chip_name} | Spatial Size: 1024x1024", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return True


def build_html_gallery() -> str:
    html_path = os.path.join(OUTPUT_DIR, "index.html")
    events = sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    
    sections = []
    for e in events:
        e_dir = os.path.join(OUTPUT_DIR, e)
        pngs = sorted([f for f in os.listdir(e_dir) if f.endswith(".png")])
        cards = []
        for p in pngs:
            rel_img = f"{e}/{p}"
            cards.append(f"""
            <div class="card">
                <h4>{e} - {p.replace('_inspection.png', '').replace('_inspection_7panel.png', '')}</h4>
                <a href="{rel_img}" target="_blank">
                    <img src="{rel_img}" loading="lazy" alt="{e} sample" />
                </a>
            </div>
            """)
        cards_html = "\n".join(cards)
        sections.append(f"""
        <div class="event-section">
            <h2>Flood Event: {e} ({len(pngs)} Inspection Samples)</h2>
            <div class="grid">{cards_html}</div>
        </div>
        """)

    all_sections = "\n".join(sections)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FloodPlanet Visual Inspection Gallery - SegFormer</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ color: #38bdf8; text-align: center; margin-bottom: 8px; }}
        p.subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 32px; }}
        .event-section {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 32px; border: 1px solid #334155; }}
        .event-section h2 {{ color: #7dd3fc; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 16px; margin-top: 16px; }}
        .card {{ background: #0f172a; border-radius: 8px; padding: 12px; border: 1px solid #334155; }}
        .card h4 {{ margin: 0 0 8px 0; font-size: 14px; color: #cbd5e1; }}
        .card img {{ width: 100%; height: auto; border-radius: 4px; display: block; border: 1px solid #475569; }}
        .card img:hover {{ border-color: #38bdf8; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>FloodPlanet Dataset Visual Inspection Gallery</h1>
    <p class="subtitle">Multi-Panel Visual Alignment, Mask Conservation, and Sensor Cross-Check Across 19 Global Flood Events</p>
    {all_sections}
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Generate and open visual inspection diagnostic panels.")
    parser.add_argument("--event", type=str, default=None, help="Inspect a specific event (e.g. Bangladesh)")
    parser.add_argument("--chip", type=str, default=None, help="Inspect a specific chip filename")
    parser.add_argument("--all", action="store_true", help="Generate inspection panels for all 19 events")
    parser.add_argument("--open", action="store_true", help="Open output figure or HTML gallery in Windows default viewer")
    args = parser.parse_args()

    manifest_csv = os.path.join(PROJECT_ROOT, "outputs", "dataset_manifest.csv")
    if not os.path.exists(manifest_csv):
        from src.data.inspect_dataset import inspect_dataset
        inspect_dataset()

    df = pd.read_csv(manifest_csv)
    target_events = [args.event] if args.event else sorted(df["event"].unique().tolist())

    last_generated = None

    print(f"\nGenerating visual inspection panels for: {target_events}...")
    for ev in target_events:
        ev_df = df[df["event"] == ev].sort_values("flood_percent").reset_index(drop=True)
        if args.chip:
            sample_chips = [args.chip]
        else:
            # Pick low, med, high samples
            n = len(ev_df)
            indices = sorted(list(set([0, n // 2, n - 1])))
            sample_chips = [ev_df.iloc[i]["filename"] for i in indices]

        for chip_name in sample_chips:
            out_name = f"{chip_name}_inspection_7panel.png"
            out_png = os.path.join(OUTPUT_DIR, ev, out_name)
            success = render_7panel(ev, chip_name, out_png)
            if success:
                print(f"  [OK] Saved 7-panel figure: {out_png}")
                last_generated = out_png

    gallery_path = build_html_gallery()
    print(f"\nUpdated HTML Gallery: {gallery_path}\n")

    if args.open:
        target_to_open = gallery_path if args.all or not last_generated else last_generated
        print(f"Opening in default viewer: {target_to_open}")
        if sys.platform == "win32":
            os.system(f'start "" "{target_to_open}"')


if __name__ == "__main__":
    main()
