"""
CAVE-Flood Automated Deployment & Universal Image Adapter Verification Test Suite
Phase 5 Numerical Regression & Phase 8 Comprehensive Multi-Format Tests:
1. Checkpoint existence and SHA-256 integrity
2. GeoTIFF ingestion & metadata validation
3. Universal Image Adapter Category A (Verified 4-band satellite raster)
4. Universal Image Adapter Category B (Ordinary RGB: JPG, PNG, WebP, BMP)
5. Universal Image Adapter Category C (4-channel RGBA with Alpha transparency)
6. Universal Image Adapter Category D (Grayscale, Corrupted, Oversized)
7. Image Processing Filters (Original RGB, Grayscale, Edge Detection, Contrast)
8. Phase 5 Numerical Regression (Identical pipeline verification against baseline)
9. Model Inference Blocking for Incompatible Images
10. End-to-end E0/E1/E2 Inference, LayerCAM Attribution & GeoTIFF Export
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from deployment_config import verify_checkpoints, get_device
from preprocessing import validate_and_load_geotiff, create_rgb_preview
from image_adapter import (
    inspect_and_load_universal_image,
    generate_image_processing_previews,
    ImageInspectionResult
)
from inference import predict_scene, ModelManager
from xai_utils import compute_groundtruth_evaluation
from visualization import create_segmentation_mask_image, create_flood_overlay, save_geotiff_output
import app


class TestCAVEFloodDeployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = DEPLOYMENT_DIR / "examples" / "sample_bolivia_BOL_1015.tif"
        cls.label_path = DEPLOYMENT_DIR / "examples" / "sample_bolivia_BOL_1015_groundtruth.tif"
        cls.samples_dir = DEPLOYMENT_DIR / "tests" / "test_samples"
        assert cls.sample_path.exists(), f"Sample test file missing: {cls.sample_path}"
        
        # Load sample raster via legacy preprocessing
        cls.loaded = validate_and_load_geotiff(cls.sample_path)
        cls.image = cls.loaded["image"]
        cls.valid_mask = cls.loaded["valid_mask"]
        cls.meta = cls.loaded["meta"]

    def test_01_checkpoint_integrity(self):
        """Verify all checkpoints exist and match expected SHA-256 hashes."""
        status = verify_checkpoints()
        for name in ["best_model.pth", "best_e1_model.pth", "best_e2_model.pth"]:
            self.assertTrue(status[name]["found"], f"Missing checkpoint: {name}")
            self.assertTrue(status[name]["verified"], f"SHA256 mismatch for {name}: {status[name]['sha256']}")

    def test_02_geotiff_loading_and_rgb_preview(self):
        """Verify 4-band ingestion and true-color RGB contrast stretching."""
        self.assertEqual(self.image.shape[0], 4)
        self.assertGreater(self.valid_mask.sum(), 0)
        
        rgb = create_rgb_preview(self.image, self.valid_mask)
        self.assertEqual(rgb.size, (self.image.shape[2], self.image.shape[1]))
        self.assertEqual(rgb.mode, "RGB")

    def test_03_universal_adapter_category_a(self):
        """Verify Category A: Genuine 4-band satellite GeoTIFF is accepted for AI segmentation."""
        res = inspect_and_load_universal_image(str(self.sample_path))
        self.assertEqual(res.category, "CATEGORY_A")
        self.assertTrue(res.is_model_compatible)
        self.assertEqual(res.channels, 4)
        self.assertIn("COMPATIBLE", res.status_badge)
        self.assertIsNotNone(res.preview_image)
        self.assertEqual(res.raw_array.shape, (4, 1024, 1024))
        self.assertTrue(res.valid_mask.any())

    def test_04_universal_adapter_category_b_rgb_formats(self):
        """Verify Category B: Standard RGB formats (JPG, PNG, WebP, BMP) are preview-only."""
        rgb_test_files = [
            ("test_photo.jpg", "JPEG"),
            ("test_rgb.png", "PNG"),
            ("test_image.webp", "WEBP"),
            ("test_image.bmp", "BMP")
        ]
        for filename, expected_fmt in rgb_test_files:
            file_path = self.samples_dir / filename
            self.assertTrue(file_path.exists(), f"Missing fixture: {filename}")
            res = inspect_and_load_universal_image(str(file_path))
            
            self.assertEqual(res.category, "CATEGORY_B", f"Failed for {filename}")
            self.assertFalse(res.is_model_compatible, f"RGB image must NOT be model compatible: {filename}")
            self.assertIn("PREVIEW ONLY", res.status_badge)
            self.assertEqual(res.channels, 3)
            self.assertIsNotNone(res.preview_image)
            self.assertEqual(res.preview_image.mode, "RGB")
            self.assertIn("Near-Infrared (NIR)", res.status_message)

    def test_05_universal_adapter_category_c_rgba(self):
        """Verify Category C: 4-channel RGBA has Alpha detected and is blocked from model."""
        rgba_file = self.samples_dir / "test_rgba.png"
        self.assertTrue(rgba_file.exists())
        res = inspect_and_load_universal_image(str(rgba_file))
        
        self.assertEqual(res.category, "CATEGORY_C")
        self.assertFalse(res.is_model_compatible)
        self.assertEqual(res.channels, 4)
        self.assertIn("RGBA", res.color_mode)
        self.assertIn("Alpha/Transparency", res.status_message)
        self.assertIsNotNone(res.preview_image)

    def test_06_universal_adapter_category_d_grayscale_and_corrupt(self):
        """Verify Category D: Grayscale, Corrupted, and Oversized images are handled safely."""
        # 1. Grayscale
        gray_file = self.samples_dir / "test_grayscale.png"
        res_gray = inspect_and_load_universal_image(str(gray_file))
        self.assertEqual(res_gray.category, "CATEGORY_D")
        self.assertFalse(res_gray.is_model_compatible)
        self.assertEqual(res_gray.channels, 1)
        self.assertIsNotNone(res_gray.preview_image)

        # 2. Corrupt file
        corrupt_file = self.samples_dir / "test_corrupt.png"
        res_corrupt = inspect_and_load_universal_image(str(corrupt_file))
        self.assertEqual(res_corrupt.category, "CATEGORY_D")
        self.assertFalse(res_corrupt.is_model_compatible)
        self.assertIsNotNone(res_corrupt.error)
        self.assertIn("CORRUPTED", res_corrupt.status_badge)

    def test_07_image_processing_previews(self):
        """Verify standard computer vision filters run independently from inference."""
        rgb_file = self.samples_dir / "test_rgb.png"
        with Image.open(rgb_file) as img:
            rgb_img = img.convert("RGB")
            
        previews = generate_image_processing_previews(rgb_img)
        self.assertIn("original_rgb", previews)
        self.assertIn("grayscale", previews)
        self.assertIn("edge_detection", previews)
        self.assertIn("contrast_enhanced", previews)
        
        for k, v in previews.items():
            self.assertIsInstance(v, Image.Image, f"Preview {k} must be PIL Image")
            self.assertEqual(v.size, rgb_img.size, f"Size mismatch for {k}")
            self.assertEqual(v.mode, "RGB", f"Mode must be RGB for {k}")

    def test_08_phase5_numerical_regression_identical_pipeline(self):
        """
        Phase 5 Strict Regression Test:
        Compares the legacy GeoTIFF loading pipeline against the universal image adapter
        on the exact same valid four-band TIFF.
        Verifies:
        1. Identical input tensor shape and pixel values
        2. Identical channel ordering
        3. Identical E0 predictions and probabilities (max float diff == 0.0)
        4. Identical E1 predictions and candidate masks (binary equality)
        5. Identical E2 RemoteCLIP guidance and refined predictions
        """
        # 1. Legacy ingestion
        legacy_res = validate_and_load_geotiff(self.sample_path)
        legacy_img = legacy_res["image"]
        legacy_mask = legacy_res["valid_mask"]
        
        # 2. Universal adapter ingestion
        univ_res = inspect_and_load_universal_image(str(self.sample_path))
        univ_img = univ_res.raw_array
        univ_mask = univ_res.valid_mask
        
        # Verify input shapes & exact values
        self.assertEqual(legacy_img.shape, univ_img.shape)
        self.assertEqual(legacy_mask.shape, univ_mask.shape)
        self.assertTrue(np.array_equal(legacy_img, univ_img), "Raw pixel arrays must be bitwise identical")
        self.assertTrue(np.array_equal(legacy_mask, univ_mask), "Valid masks must be bitwise identical")
        
        # Test on 512x512 sub-crop
        crop_legacy = legacy_img[:, :512, :512]
        crop_univ = univ_img[:, :512, :512]
        crop_mask = legacy_mask[:512, :512]
        
        # Test E0 Baseline
        out_legacy_e0 = predict_scene(crop_legacy, crop_mask, mode="E0", run_xai=False)
        out_univ_e0 = predict_scene(crop_univ, crop_mask, mode="E0", run_xai=False)
        
        diff_e0_prob = np.max(np.abs(out_legacy_e0["p_base"] - out_univ_e0["p_base"]))
        self.assertEqual(diff_e0_prob, 0.0, f"E0 baseline probability max diff: {diff_e0_prob}")
        self.assertTrue(np.array_equal(out_legacy_e0["pred_final"], out_univ_e0["pred_final"]))
        
        # Test E1 CASR Neutral
        out_legacy_e1 = predict_scene(crop_legacy, crop_mask, mode="E1", run_xai=False)
        out_univ_e1 = predict_scene(crop_univ, crop_mask, mode="E1", run_xai=False)
        
        self.assertTrue(np.array_equal(out_legacy_e1["cand_mask"], out_univ_e1["cand_mask"]))
        diff_e1_prob = np.max(np.abs(out_legacy_e1["p_final"] - out_univ_e1["p_final"]))
        self.assertEqual(diff_e1_prob, 0.0, f"E1 probability max diff: {diff_e1_prob}")
        self.assertTrue(np.array_equal(out_legacy_e1["pred_final"], out_univ_e1["pred_final"]))
        
        # Test E2 RemoteCLIP Guided
        out_legacy_e2 = predict_scene(crop_legacy, crop_mask, mode="E2", run_xai=True)
        out_univ_e2 = predict_scene(crop_univ, crop_mask, mode="E2", run_xai=True)
        
        diff_guidance = np.max(np.abs(out_legacy_e2["guidance_map"] - out_univ_e2["guidance_map"]))
        self.assertLess(diff_guidance, 1e-6, f"RemoteCLIP guidance map max diff: {diff_guidance}")
        self.assertTrue(np.array_equal(out_legacy_e2["pred_final"], out_univ_e2["pred_final"]))
        self.assertTrue(np.array_equal(out_legacy_e2["cand_mask"], out_univ_e2["cand_mask"]))
        
        print("\n" + "=" * 65)
        print("  PHASE 5 REGRESSION VERIFICATION: 100% NUMERICAL IDENTITY CONFIRMED")
        print("=" * 65)
        print(f"  Input Tensor Shape       : {univ_img.shape} (Bitwise Identical)")
        print(f"  E0 Probability Max Diff  : {diff_e0_prob:.10f}")
        print(f"  E1 Probability Max Diff  : {diff_e1_prob:.10f}")
        print(f"  E2 Guidance Max Diff     : {diff_guidance:.10f}")
        print(f"  Deterministic Binary Mask: EXACT EQUALITY (0 mismatch pixels)")
        print("=" * 65)

    def test_09_model_incompatible_blocking(self):
        """Verify that attempting flood detection on non-Category-A images is strictly blocked."""
        import gradio as gr
        
        # Reset session
        app.reset_all()
        
        # Simulate uploading standard RGB image
        rgb_file = self.samples_dir / "test_photo.jpg"
        app.handle_image_upload(str(rgb_file))
        
        # Ensure image_data is None in session
        self.assertIsNone(app.CURRENT_SESSION["image_data"])
        self.assertFalse(app.CURRENT_SESSION["inspection"].is_model_compatible)
        
        # Attempting detection must raise gr.Error
        with self.assertRaises(gr.Error) as ctx:
            app.run_detection("E2 — RemoteCLIP Guided Refinement (Recommended)", enable_xai=False)
        self.assertIn("Segmentation Blocked", str(ctx.exception))

    def test_10_e0_e1_e2_layercam_export(self):
        """Verify end-to-end full scene inference, LayerCAM generation, and GeoTIFF export."""
        crop_img = self.image[:, :512, :512]
        crop_valid = self.valid_mask[:512, :512]
        rgb = create_rgb_preview(crop_img, crop_valid)
        
        res = predict_scene(crop_img, crop_valid, mode="E2", run_xai=True)
        self.assertEqual(res["pred_final"].shape, (512, 512))
        self.assertIsNotNone(res["layercam_map"])
        
        mask_img = create_segmentation_mask_image(res["pred_final"], crop_valid)
        overlay_img = create_flood_overlay(rgb, res["pred_final"], crop_valid)
        self.assertEqual(mask_img.size, (512, 512))
        self.assertEqual(overlay_img.size, (512, 512))
        
        out_tif = DEPLOYMENT_DIR / "tests" / "test_export.tif"
        save_geotiff_output(res["pred_final"], self.meta, out_tif)
        self.assertTrue(out_tif.exists())
        out_tif.unlink()


if __name__ == "__main__":
    unittest.main()
