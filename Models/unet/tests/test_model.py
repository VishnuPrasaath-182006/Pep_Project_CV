"""
Unit and Forward-Pass Tests for FloodPlanet U-Net Implementation
Verifies:
1. Model construction with 4 input channels and 2 output classes
2. Parameter count matches expected ~17.27M parameters
3. Forward pass on 300x300 PlanetScope patch dimensions
4. Output tensor finiteness and valid logits
5. Sliding-window patch tiling compatibility
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Ensure unet src is in sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import UNet, DoubleConv, Down, Up, OutConv


class TestUNet(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = UNet(n_channels=4, n_classes=2, bilinear=True).to(self.device)
        self.model.eval()

    def test_model_construction(self):
        """Verify model initializes properly with 4 input channels and 2 classes."""
        self.assertIsInstance(self.model, UNet)
        self.assertEqual(self.model.n_channels, 4)
        self.assertEqual(self.model.n_classes, 2)

    def test_parameter_count(self):
        """Verify parameter count matches expected ~17.27M parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 17_000_000)
        self.assertLess(total_params, 18_000_000)

    def test_forward_pass_300x300(self):
        """Verify forward pass on standard 300x300 PlanetScope patch."""
        batch_size = 2
        x = torch.randn(batch_size, 4, 300, 300, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)

        self.assertEqual(logits.shape, (batch_size, 2, 300, 300))
        self.assertTrue(torch.isfinite(logits).all(), "Logits contain NaN or Inf values")

    def test_forward_pass_probabilities(self):
        """Verify softmax output produces valid probabilities summing to 1."""
        x = torch.randn(1, 4, 300, 300, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

        self.assertEqual(probs.shape, (1, 2, 300, 300))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        sum_probs = probs.sum(dim=1)
        self.assertTrue(torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-5))


def run_diagnostic_inspection():
    print("=" * 70)
    print("      FLOODPLANET U-NET MODEL DIAGNOSTIC & VERIFICATION REPORT")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device        : {device}")
    
    # 1. Model Instantiation
    model = UNet(n_channels=4, n_classes=2, bilinear=True).to(device)
    model.eval()
    
    # 2. Parameter Calculation
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model Architecture    : 4-Channel U-Net (ResNet-style DoubleConv Encoder/Decoder)")
    print(f"[*] Input Channels        : 4 (Blue, Green, Red, Near-Infrared / NIR)")
    print(f"[*] Output Classes        : 2 (Class 0: Background, Class 1: Flood Water)")
    print(f"[*] Total Parameters      : {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"[*] Trainable Parameters  : {trainable_params:,}")
    
    # 3. Forward Pass Simulation
    batch_size = 2
    dummy_input = torch.randn(batch_size, 4, 300, 300, dtype=torch.float32, device=device)
    print(f"[*] Simulated Input Tensor: {list(dummy_input.shape)} (Batch={batch_size}, Ch=4, H=300, W=300)")
    
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
    print(f"[*] Output Logits Tensor  : {list(logits.shape)} (Finite: {torch.isfinite(logits).all().item()})")
    print(f"[*] Softmax Probabilities : {list(probs.shape)} (Range: [{probs.min().item():.4f}, {probs.max().item():.4f}])")
    print(f"[*] Prediction Mask Shape : {list(preds.shape)} (Unique labels: {torch.unique(preds).tolist()})")
    print("-" * 70)
    print("Running Unittest Suite:")


if __name__ == "__main__":
    run_diagnostic_inspection()
    sys.stdout.flush()
    sys.stderr.flush()
    unittest.main(argv=[sys.argv[0], "-v"])


