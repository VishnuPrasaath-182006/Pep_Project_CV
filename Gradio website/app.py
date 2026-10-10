"""
CAVE-Flood Public Demonstration Web Application
Confidence-Aware Vision-Language-Guided Residual Refinement for Flood Inundation Segmentation
Built with Gradio Blocks for deployment to Hugging Face Spaces.
"""

import os
import sys
import tempfile
from pathlib import Path
import gradio as gr
import numpy as np
from PIL import Image

DEPLOYMENT_DIR = Path(__file__).resolve().parent
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from deployment_config import (
    ASSETS_DIR,
    EXAMPLES_DIR,
    verify_checkpoints,
    get_device
)
from preprocessing import (
    validate_and_load_geotiff,
    create_rgb_preview,
    validate_and_load_groundtruth
)
from image_adapter import (
    inspect_and_load_universal_image,
    generate_image_processing_previews,
    SUPPORTED_EXTENSIONS,
    ImageInspectionResult
)
from inference import predict_scene, ModelManager
from visualization import (
    create_segmentation_mask_image,
    create_flood_overlay,
    create_comparison_panel,
    create_xai_dashboard_image,
    save_geotiff_output
)
from xai_utils import compute_groundtruth_evaluation

# Load custom CSS
css_path = ASSETS_DIR / "custom.css"
custom_css = ""
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        custom_css = f.read()

# Model options and descriptions
MODEL_OPTIONS = [
    "E2 — RemoteCLIP Guided Refinement (Recommended)",
    "E1 — CASR Refinement (Neutral Guidance)",
    "E0 — UNet-Mamba Baseline (Frozen Backbone)"
]

MODEL_DESCRIPTIONS = {
    "E2 — RemoteCLIP Guided Refinement (Recommended)": (
        "**E2 Configuration**: Evaluates the frozen UNet-Mamba backbone refined by the 12,035-parameter CASR adapter head, "
        "guided by satellite-domain vision-language cosine similarity embeddings from frozen RemoteCLIP (ViT-B/32). "
        "Selects the top 25% most uncertain boundary regions while maintaining exact zero-residual invariance (ΔZ ≡ 0) "
        "across the remaining 75% of confident interior pixels."
    ),
    "E1 — CASR Refinement (Neutral Guidance)": (
        "**E1 Configuration**: Evaluates the frozen UNet-Mamba backbone with CASR residual adaptation using neutral guidance (G=0). "
        "Tests confidence-aware boundary modulation based purely on spatial entropy and gradient cues without language embeddings."
    ),
    "E0 — UNet-Mamba Baseline (Frozen Backbone)": (
        "**E0 Configuration**: Evaluates the standalone 14.29M-parameter UNet-Mamba foundation model. "
        "Bypasses residual adaptation entirely, returning the raw baseline probability predictions."
    )
}

def get_mode_code(selected_text):
    if "E0" in selected_text:
        return "E0"
    if "E1" in selected_text:
        return "E1"
    return "E2"


# Application State Cache for current active session
CURRENT_SESSION = {
    "image_data": None,
    "valid_mask": None,
    "meta": None,
    "rgb_preview": None,
    "inspection": None,
    "gt_mask": None,
    "pred_mask": None,
    "pred_base": None,
    "temp_dir": tempfile.TemporaryDirectory()
}


def handle_image_upload(file_obj, *args):
    """
    Universally inspects uploaded image (TIFF, GeoTIFF, PNG, JPG, WebP, BMP),
    determines spectral and model compatibility, and prepares previews.
    """
    if file_obj is None:
        CURRENT_SESSION["image_data"] = None
        CURRENT_SESSION["valid_mask"] = None
        CURRENT_SESSION["meta"] = None
        CURRENT_SESSION["rgb_preview"] = None
        CURRENT_SESSION["inspection"] = None
        return (
            None,
            gr.update(value="", visible=False),
            "",
            None,
            None,
            None,
            None,
            gr.update(interactive=False),
            gr.update(visible=False)
        )
        
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    insp = inspect_and_load_universal_image(file_path)
    CURRENT_SESSION["inspection"] = insp
    
    # 1. Format compatibility callout card
    if insp.category == "CATEGORY_A":
        # Full 4-band satellite compatibility
        CURRENT_SESSION["image_data"] = insp.raw_array
        CURRENT_SESSION["valid_mask"] = insp.valid_mask
        CURRENT_SESSION["meta"] = insp.meta
        CURRENT_SESSION["rgb_preview"] = insp.preview_image
        
        compat_html = (
            f'<div class="status-badge-compatible">'
            f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">'
            f'<span style="font-size: 16px;">🟢</span>'
            f'<strong style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.03em;">Compatible: Four-Band Model Inference Available</strong>'
            f'</div>'
            f'<div style="font-size: 13px; line-height: 1.4;">{insp.status_message}</div>'
            f'</div>'
        )
        is_interactive = True
        
    elif insp.category in ["CATEGORY_B", "CATEGORY_C"]:
        # Preview only: ordinary RGB or RGBA transparency
        CURRENT_SESSION["image_data"] = None
        CURRENT_SESSION["valid_mask"] = insp.valid_mask
        CURRENT_SESSION["meta"] = insp.meta
        CURRENT_SESSION["rgb_preview"] = insp.preview_image
        
        badge_title = "Preview Only: Segmentation Requires 4-Band Satellite Data"
        compat_html = (
            f'<div class="status-badge-preview">'
            f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">'
            f'<span style="font-size: 16px;">🟡</span>'
            f'<strong style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.03em;">{badge_title}</strong>'
            f'</div>'
            f'<div style="font-size: 13px; line-height: 1.4;">{insp.status_message}</div>'
            f'</div>'
        )
        is_interactive = False
        
    else:
        # Unsupported or corrupted
        CURRENT_SESSION["image_data"] = None
        CURRENT_SESSION["valid_mask"] = None
        CURRENT_SESSION["meta"] = None
        CURRENT_SESSION["rgb_preview"] = insp.preview_image
        
        compat_html = (
            f'<div class="status-badge-unsupported">'
            f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">'
            f'<span style="font-size: 16px;">🔴</span>'
            f'<strong style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.03em;">Unsupported: File Cannot Be Processed</strong>'
            f'</div>'
            f'<div style="font-size: 13px; line-height: 1.4;">{insp.status_message}</div>'
            f'</div>'
        )
        is_interactive = False
        
    # 2. Metadata table
    info_markdown = (
        f"| Attribute | Property |\n"
        f"| :--- | :--- |\n"
        f"| **Detected Format** | `{insp.format_name}` |\n"
        f"| **Dimensions** | `{insp.dimensions_str}` |\n"
        f"| **Sensor Channels** | `{insp.channels}` |\n"
        f"| **Color Mode** | `{insp.color_mode}` |\n"
        f"| **Bit Depth** | `{insp.bit_depth}` |\n"
        f"| **Projection / CRS** | `{insp.crs}` |\n"
        f"| **Valid Data Area** | `{insp.valid_coverage_pct}%` |\n"
        f"| **File Size** | `{insp.file_size_mb} MB` |"
    )
    
    # 3. Image processing previews
    p_orig, p_gray, p_edge, p_contrast = None, None, None, None
    if insp.preview_image is not None:
        previews = generate_image_processing_previews(insp.preview_image)
        p_orig = previews.get("original_rgb")
        p_gray = previews.get("grayscale")
        p_edge = previews.get("edge_detection")
        p_contrast = previews.get("contrast_enhanced")
        
    return (
        insp.preview_image,
        gr.update(value=compat_html, visible=True),
        info_markdown,
        p_orig,
        p_gray,
        p_edge,
        p_contrast,
        gr.update(interactive=is_interactive),
        gr.update(visible=False)
    )


def handle_gt_upload(file_obj):
    """
    Validates optional ground-truth mask.
    """
    if file_obj is None:
        CURRENT_SESSION["gt_mask"] = None
        return "No ground truth provided (Normal operational inference)."
        
    if CURRENT_SESSION["image_data"] is None:
        return "Upload a compatible 4-band satellite image first to check ground-truth alignment."
        
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    try:
        _, h, w = CURRENT_SESSION["image_data"].shape
        gt_mask = validate_and_load_groundtruth(file_path, (h, w))
        CURRENT_SESSION["gt_mask"] = gt_mask
        eval_px = int((CURRENT_SESSION["valid_mask"] & (gt_mask != -1)).sum())
        return f"Ground truth mask verified. {eval_px:,} valid evaluation pixels ready for validation metrics."
    except Exception as e:
        CURRENT_SESSION["gt_mask"] = None
        return f"Ground-truth format error: {str(e)}"


def run_detection(model_choice, enable_xai, progress=gr.Progress()):
    """
    Runs end-to-end flood segmentation inference and renders all outputs.
    Strictly blocked if uploaded imagery is not verified 4-band satellite data.
    """
    insp = CURRENT_SESSION.get("inspection")
    if insp is None or not insp.is_model_compatible or CURRENT_SESSION["image_data"] is None:
        fmt = insp.format_name if insp else "Unknown"
        cat = insp.category if insp else "None"
        raise gr.Error(
            f"Segmentation Blocked ({fmt}, {cat}): "
            f"CAVE-Flood requires genuine four-band PlanetScope satellite imagery (Blue, Green, Red, Near-Infrared). "
            f"Ordinary RGB/RGBA images cannot produce scientific flood segmentation. "
            f"Please upload a compatible 4-band GeoTIFF."
        )
        
    mode = get_mode_code(model_choice)
    img_arr = CURRENT_SESSION["image_data"]
    valid_mask = CURRENT_SESSION["valid_mask"]
    rgb_preview = CURRENT_SESSION["rgb_preview"]
    meta = CURRENT_SESSION["meta"]
    
    progress(0.1, desc="Preparing satellite patches and loading model weights...")
    
    try:
        # Run inference
        progress(0.3, desc=f"Executing {mode} segmentation pass...")
        res = predict_scene(img_arr, valid_mask, mode=mode, run_xai=enable_xai)
        
        progress(0.7, desc="Rendering high-resolution masks and overlays...")
        pred_final = res["pred_final"]
        pred_base = res["pred_base"]
        stats = res["stats"]
        
        CURRENT_SESSION["pred_mask"] = pred_final
        CURRENT_SESSION["pred_base"] = pred_base
        
        # 1. Output Visualizations
        mask_img = create_segmentation_mask_image(pred_final, valid_mask)
        overlay_img = create_flood_overlay(rgb_preview, pred_final, valid_mask, alpha=0.45)
        comp_img = create_comparison_panel(
            rgb_preview, pred_base, pred_final, valid_mask,
            base_title="E0 UNet-Mamba",
            ref_title=f"{mode} Refined"
        )
        
        # 2. XAI Dashboard
        xai_img = None
        if enable_xai:
            progress(0.85, desc="Computing LayerCAM and spatial diagnostic insights...")
            xai_img = create_xai_dashboard_image(
                entropy_map=res["entropy_map"],
                cand_mask=res["cand_mask"],
                gate_map=res["gate_map"],
                guidance_map=res["guidance_map"],
                prob_delta=res["prob_delta"],
                layercam_map=res["layercam_map"],
                valid_mask=valid_mask
            )
            
        # 3. Ground-truth evaluation (if GT was supplied)
        gt_metrics_text = "Ground-truth evaluation was not requested or no mask was supplied."
        gt_error_img = None
        if CURRENT_SESSION["gt_mask"] is not None:
            metrics, err_img = compute_groundtruth_evaluation(pred_final, CURRENT_SESSION["gt_mask"], valid_mask)
            gt_error_img = err_img
            gt_metrics_text = (
                f"### Ground-Truth Validation Metrics\n"
                f"- **Flood IoU**: `{metrics['flood_iou']}%`\n"
                f"- **Flood Dice / F1 Score**: `{metrics['flood_dice']}%`\n"
                f"- **Precision**: `{metrics['precision']}%`\n"
                f"- **Recall / Sensitivity**: `{metrics['recall']}%`\n"
                f"- **Specificity**: `{metrics['specificity']}%`\n"
                f"- **Pixel Accuracy**: `{metrics['pixel_accuracy']}%`\n\n"
                f"*Physical Confusion Pixels: TP={metrics['tp']:,} | FP={metrics['fp']:,} | FN={metrics['fn']:,} | TN={metrics['tn']:,}*"
            )
            
        # 4. Save downloadable outputs
        temp_dir = Path(CURRENT_SESSION["temp_dir"].name)
        mask_path = temp_dir / "flood_segmentation_mask.png"
        mask_img.save(mask_path)
        
        overlay_path = temp_dir / "flood_satellite_overlay.png"
        overlay_img.save(overlay_path)
        
        geotiff_path = temp_dir / "predicted_flood_mask.tif"
        save_geotiff_output(pred_final, meta, geotiff_path)
        
        # Format Statistics Scorecard
        stats_markdown = (
            f"### Segmentation Summary ({mode})\n"
            f"- **Predicted Flood Extent**: **`{stats['refined_flood_pct']}%`** ({stats['refined_flood_pixels']:,} pixels)\n"
            f"- **Baseline (E0) Extent**: `{stats['base_flood_pct']}%` ({stats['base_flood_pixels']:,} pixels)\n"
            f"- **CASR Boundary Corrections**: `+{stats['added_flood_pixels']:,}` flood pixels added | `-{stats['suppressed_pixels']:,}` false alarms suppressed\n"
            f"- **Boundary Region Coverage**: `{stats['candidate_coverage_pct']}%` of scene received residual adaptation (remaining 75% invariant)"
        )
        
        progress(1.0, desc="Done!")
        return (
            rgb_preview,
            mask_img,
            overlay_img,
            comp_img,
            stats_markdown,
            xai_img,
            gt_error_img,
            gt_metrics_text,
            str(mask_path),
            str(overlay_path),
            str(geotiff_path),
            gr.update(visible=True)
        )
        
    except Exception as e:
        raise gr.Error(f"Inference Pipeline Error: {str(e)}")


def reset_all():
    """Resets user interface state and temporary session cache."""
    CURRENT_SESSION["image_data"] = None
    CURRENT_SESSION["valid_mask"] = None
    CURRENT_SESSION["meta"] = None
    CURRENT_SESSION["rgb_preview"] = None
    CURRENT_SESSION["inspection"] = None
    CURRENT_SESSION["gt_mask"] = None
    CURRENT_SESSION["pred_mask"] = None
    CURRENT_SESSION["pred_base"] = None
    return (
        None,                             # input_file
        None,                             # gt_file
        "No ground truth provided (Normal operational inference).", # gt_status
        gr.update(value=MODEL_OPTIONS[0]),# model_selector
        True,                             # enable_xai_checkbox
        None,                             # rgb_preview_display
        gr.update(value="", visible=False),# compat_status
        "",                               # meta_display
        None,                             # proc_orig
        None,                             # proc_gray
        None,                             # proc_edge
        None,                             # proc_contrast
        gr.update(interactive=False),     # run_btn
        None,                             # orig_card_display
        None,                             # mask_display
        None,                             # overlay_display
        None,                             # comp_display
        "",                               # stats_display
        None,                             # xai_dashboard_display
        None,                             # gt_error_display
        "Ground truth not uploaded.",     # gt_metrics_display
        None,                             # dl_mask
        None,                             # dl_overlay
        None,                             # dl_geotiff
        gr.update(visible=False)          # results_container
    )


# ---------------------------------------------------------------------------
# Gradio Blocks Application Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="CAVE-Flood - AI Flood Segmentation Demo") as demo:
    
    # SECTION 1: HEADER
    with gr.Row():
        with gr.Column(elem_classes=["header-box"]):
            gr.HTML(
                """
                <span class="header-badge">Research Demo</span>
                <h1 class="header-title">CAVE-Flood</h1>
                <div class="header-subtitle">AI-Powered Flood Inundation Segmentation</div>
                <p class="header-desc">
                    Explore satellite flood detection using UNet-Mamba, confidence-aware refinement and vision-language guidance.
                    Combines a frozen <strong>UNet-Mamba</strong> foundation backbone, the parameter-efficient 
                    <strong>Confidence-Aware Semantic Residual (CASR)</strong> adapter head, and zero-shot 
                    vision-language guidance from <strong>RemoteCLIP</strong>.
                </p>
                """
            )
            
    # SECTION 2: INPUTS & CONFIGURATION
    with gr.Row():
        # Left Column: Upload & Options
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["card-section"]):
                gr.Markdown("### 1. Upload Satellite Image", elem_classes=["section-title"])
                gr.Markdown(
                    "Upload GeoTIFF, TIFF, PNG, JPG, JPEG, WebP, or BMP imagery. "
                    "Flood detection requires compatible four-band satellite data.",
                    elem_classes=["section-desc"]
                )
                
                input_file = gr.File(
                    label="Upload Satellite Image / Imagery Raster",
                    file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                    file_count="single",
                    type="filepath"
                )
                
                with gr.Accordion("Optional: Upload Ground-Truth Mask", open=False):
                    gt_file = gr.File(
                        label="Upload Ground-Truth Mask (.tif) for Error Analysis",
                        file_types=[".tif", ".tiff"],
                        file_count="single",
                        type="filepath"
                    )
                    gt_status = gr.Markdown("No ground truth provided (Normal operational inference).")
                    
            with gr.Group(elem_classes=["card-section"]):
                gr.Markdown("### 2. Model Selection", elem_classes=["section-title"])
                model_selector = gr.Radio(
                    choices=MODEL_OPTIONS,
                    value=MODEL_OPTIONS[0],
                    label="Select Architecture Paradigm",
                    elem_classes=["model-radio"]
                )
                model_info = gr.Markdown(MODEL_DESCRIPTIONS[MODEL_OPTIONS[0]])
                
                enable_xai_checkbox = gr.Checkbox(
                    label="Compute Explainable AI (LayerCAM Heatmap & Entropy)",
                    value=True,
                    info="Generates predictive entropy, candidate mask, gate activations, and LayerCAM attribution."
                )
                
                with gr.Row():
                    run_btn = gr.Button("Detect Flood Regions", variant="primary", elem_classes=["primary-btn"], interactive=False)
                    reset_btn = gr.Button("Reset", variant="secondary")

        # Right Column: Satellite Image Preview, Compatibility Status & Processing Previews
        with gr.Column(scale=6):
            with gr.Group(elem_classes=["card-section"]):
                gr.Markdown("### Image Preview & Verification", elem_classes=["section-title"])
                
                compat_status = gr.HTML("", visible=False)
                
                rgb_preview_display = gr.Image(
                    label="Optical Preview",
                    type="pil",
                    elem_classes=["preview-box"]
                )
                meta_display = gr.Markdown("", label="Imagery Metadata")
                
                # Image Processing Preview Accordion
                with gr.Accordion("Image Processing Preview (Standard Computer Vision Filters)", open=False):
                    gr.Markdown(
                        "*Basic optical filters (Grayscale luminance, Sobel boundary gradient, Dynamic contrast stretch). "
                        "Strictly separated from AI flood segmentation.*"
                    )
                    with gr.Row():
                        proc_orig = gr.Image(label="1. Original Optical Image", type="pil", elem_classes=["preview-box"])
                        proc_gray = gr.Image(label="2. Grayscale (Luminance)", type="pil", elem_classes=["preview-box"])
                    with gr.Row():
                        proc_edge = gr.Image(label="3. Edge Detection (Gradient Boundaries)", type="pil", elem_classes=["preview-box"])
                        proc_contrast = gr.Image(label="4. Contrast Enhanced (Dynamic Stretch)", type="pil", elem_classes=["preview-box"])

    # SECTION 3: PREDICTION RESULTS
    with gr.Row(visible=False) as results_container:
        with gr.Column():
            with gr.Group(elem_classes=["card-section"]):
                gr.Markdown("## Flood Detection Results", elem_classes=["section-title"])
                
                stats_display = gr.Markdown("", elem_classes=["stat-pill"])
                
                with gr.Tabs():
                    with gr.TabItem("Flood Segmentation Outputs (3 Clean Cards)"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                orig_card_display = gr.Image(
                                    label="Original Satellite Image (True-Color RGB)",
                                    type="pil",
                                    elem_classes=["preview-box"]
                                )
                            with gr.Column(scale=1):
                                mask_display = gr.Image(
                                    label="Predicted Flood Segmentation Mask",
                                    type="pil",
                                    elem_classes=["preview-box"]
                                )
                            with gr.Column(scale=1):
                                overlay_display = gr.Image(
                                    label="Flood Overlay on Satellite Image (Transparent Blue)",
                                    type="pil",
                                    elem_classes=["preview-box"]
                                )
                                
                    with gr.TabItem("Multi-Model Comparative Analysis"):
                        comp_display = gr.Image(
                            label="Comparative Transition Panel (RGB, Baseline E0, Refined E1/E2, CASR Delta)",
                            type="pil",
                            elem_classes=["preview-box"]
                        )
                        
                with gr.Row():
                    dl_mask = gr.DownloadButton("Download Mask (PNG)", variant="secondary")
                    dl_overlay = gr.DownloadButton("Download Overlay (PNG)", variant="secondary")
                    dl_geotiff = gr.DownloadButton("Download Georeferenced GeoTIFF (.tif)", variant="secondary")

    # SECTION 4: EXPLAINABLE AI & DIAGNOSTICS (COLLAPSIBLE)
    with gr.Accordion("Explore AI Insights", open=False):
        with gr.Group(elem_classes=["card-section"]):
            gr.Markdown(
                "### Spatial Uncertainty, Gating Dynamics & LayerCAM Attribution\n"
                "*Inspect how CAVE-Flood isolates boundary uncertainty and modulates residual refinement.*\n\n"
                "> **LayerCAM Interpretation**: LayerCAM heatmaps explain the parameter-efficient CASR adapter head's "
                "residual feature representations on boundary blocks, specifically attributing spatial gradient energy to "
                "residual adjustments rather than the entire end-to-end frozen baseline prediction."
            )
            xai_dashboard_display = gr.Image(
                label="6-Panel Diagnostic Dashboard (Entropy, Candidate Mask, Gate Map, Guidance, Delta P, LayerCAM)",
                type="pil",
                elem_classes=["preview-box"]
            )
            
            with gr.Row():
                with gr.Column():
                    gt_metrics_display = gr.Markdown("Ground truth not uploaded.")
                with gr.Column():
                    gt_error_display = gr.Image(
                        label="Confusion Error Map (Green: TP, Red: FP, Yellow: FN, Dark: TN)",
                        type="pil",
                        elem_classes=["preview-box"]
                    )

    # SECTION 5: READY-TO-TEST EXAMPLES
    with gr.Group(elem_classes=["card-section"]):
        gr.Markdown("### Ready-to-Test Sample Satellite Chips & Optical Images", elem_classes=["section-title"])
        gr.Markdown("Click a sample chip below to automatically load and inspect compatible 4-band PlanetScope imagery or ordinary RGB images.")
        
        sample_bolivia = str(EXAMPLES_DIR / "sample_bolivia_BOL_1015.tif")
        sample_colombia = str(EXAMPLES_DIR / "sample_colombia_COL_23_11.tif")
        sample_rgb = str(EXAMPLES_DIR / "sample_optical_rgb.jpg")
        
        example_inputs = []
        if os.path.exists(sample_bolivia):
            example_inputs.append([sample_bolivia, "E2 — RemoteCLIP Guided Refinement (Recommended)"])
        if os.path.exists(sample_colombia):
            example_inputs.append([sample_colombia, "E2 — RemoteCLIP Guided Refinement (Recommended)"])
        if os.path.exists(sample_rgb):
            example_inputs.append([sample_rgb, "E2 — RemoteCLIP Guided Refinement (Recommended)"])
            
        if example_inputs:
            gr.Examples(
                examples=example_inputs,
                inputs=[input_file, model_selector],
                fn=handle_image_upload,
                outputs=[
                    rgb_preview_display,
                    compat_status,
                    meta_display,
                    proc_orig,
                    proc_gray,
                    proc_edge,
                    proc_contrast,
                    run_btn,
                    results_container
                ],
                label="Sample Satellite Imagery & Optical Images"
            )

    # Footer note
    gr.HTML(
        """
        <div class="footer-note">
            <strong>CAVE-Flood — Research Demonstration</strong> • 
            <a href="https://github.com/VishnuPrasaath-182006/Pep_Project_CV" target="_blank" style="color: #0284c7; text-decoration: none; font-weight: 600;">GitHub Repository</a><br>
            <span style="font-size: 11.5px; color: #94a3b8; margin-top: 4px; display: inline-block;">Research prototype. Not intended for emergency decision-making.</span>
        </div>
        """
    )

    # -----------------------------------------------------------------------
    # Interactive Event Bindings
    # -----------------------------------------------------------------------
    input_file.change(
        fn=handle_image_upload,
        inputs=[input_file],
        outputs=[
            rgb_preview_display,
            compat_status,
            meta_display,
            proc_orig,
            proc_gray,
            proc_edge,
            proc_contrast,
            run_btn,
            results_container
        ]
    )
    
    gt_file.change(
        fn=handle_gt_upload,
        inputs=[gt_file],
        outputs=[gt_status]
    )
    
    model_selector.change(
        fn=lambda choice: MODEL_DESCRIPTIONS.get(choice, ""),
        inputs=[model_selector],
        outputs=[model_info]
    )
    
    run_btn.click(
        fn=run_detection,
        inputs=[model_selector, enable_xai_checkbox],
        outputs=[
            orig_card_display,
            mask_display,
            overlay_display,
            comp_display,
            stats_display,
            xai_dashboard_display,
            gt_error_display,
            gt_metrics_display,
            dl_mask,
            dl_overlay,
            dl_geotiff,
            results_container
        ]
    )
    
    reset_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            input_file,
            gt_file,
            gt_status,
            model_selector,
            enable_xai_checkbox,
            rgb_preview_display,
            compat_status,
            meta_display,
            proc_orig,
            proc_gray,
            proc_edge,
            proc_contrast,
            run_btn,
            orig_card_display,
            mask_display,
            overlay_display,
            comp_display,
            stats_display,
            xai_dashboard_display,
            gt_error_display,
            gt_metrics_display,
            dl_mask,
            dl_overlay,
            dl_geotiff,
            results_container
        ]
    )


if __name__ == "__main__":
    device = get_device()
    print("=" * 60)
    print("  LAUNCHING CAVE-FLOOD GRADIO PUBLIC RESEARCH DEMO")
    print("=" * 60)
    print(f"[*] Compute Device Detected : {device}")
    
    # Verify model checkpoints
    ckpt_status = verify_checkpoints()
    for name, s in ckpt_status.items():
        if isinstance(s, dict) and "verified" in s:
            status_str = "VERIFIED (SHA-256 MATCH)" if s['verified'] else ("FOUND" if s['found'] else "MISSING")
            print(f"[*] Checkpoint {name:<20} : {status_str}")
            
    print("=" * 60)
    print("Starting Gradio Web Server...")
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, css=custom_css, theme=gr.themes.Base())

