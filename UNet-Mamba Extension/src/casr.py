"""
CASR: Confidence-Aware Spatial Refinement Module & CAVE-Flood Model Wrapper
Implements:
1. CASRAdapter: Small CNN with two 3x3 conv blocks (~32 channels), 1-channel sigmoid gate G,
   and 2-logit residual head DeltaZ (zero-initialized).
2. Refinement update: Z_final = Z_base + lambda * C * G * DeltaZ
3. CAVEFloodModel: Non-destructive wrapper around frozen baseline UNetMamba with CASR bypass switch.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


class CASRAdapter(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=32, num_classes=2, lambda_res=1.0):
        super().__init__()
        self.lambda_res = float(lambda_res)
        self.num_classes = num_classes
        
        # Two 3x3 Conv Blocks with BatchNorm and ReLU
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1-channel Sigmoid Gate
        self.gate_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        
        # 2-logit Residual Head
        self.res_head = nn.Conv2d(hidden_channels, num_classes, kernel_size=1, bias=True)
        
        # Zero-initialize the final residual head for numerical baseline equivalence
        self._init_weights()

    def _init_weights(self):
        # Standard kaiming initialization for feature blocks
        for m in [self.block1, self.block2]:
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    
        # Bias gate slightly positive (e.g. initial gate ~ 0.5)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)
        
        # Strictly zero-initialize residual head: DeltaZ = 0 at initialization
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)

    def forward(self, features, candidate_mask):
        """
        features: (B, 9, H, W)
        candidate_mask: (B, 1, H, W)
        Returns:
            delta_z: (B, 2, H, W)
            gate: (B, 1, H, W)
            residual_update: lambda * C * G * delta_z (B, 2, H, W)
        """
        x = self.block1(features)
        x = self.block2(x)
        
        gate = torch.sigmoid(self.gate_conv(x))       # (B, 1, H, W) in [0, 1]
        delta_z = self.res_head(x)                    # (B, 2, H, W)
        
        # Gated candidate residual update: lambda * C * G * DeltaZ
        residual_update = self.lambda_res * candidate_mask * gate * delta_z
        return delta_z, gate, residual_update


class CAVEFloodModel(nn.Module):
    """
    Non-destructive wrapper around frozen base UNetMamba with CASR refinement adapter.
    """
    def __init__(self, base_model, in_channels=9, hidden_channels=32, lambda_res=1.0):
        super().__init__()
        self.base_model = base_model
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        
        # CASR trainable adapter
        self.casr = CASRAdapter(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=2,
            lambda_res=lambda_res
        )
        self.bypass_casr = False

    def forward(self, img_norm, candidate_features=None, candidate_mask=None):
        """
        img_norm: (B, 4, H, W) normalized 4-band input
        candidate_features: (B, 9, H, W) [img(4), flood_prob(1), entropy(1), grad(1), cand(1), vlm(1)]
        candidate_mask: (B, 1, H, W)
        """
        # Baseline forward pass
        with torch.no_grad():
            if hasattr(self.base_model, 'training') and self.base_model.training:
                self.base_model.eval()
            z_base = self.base_model(img_norm)
            if isinstance(z_base, (tuple, list)):
                z_base = z_base[0]  # Take main segmentation logits
                
        if self.bypass_casr or candidate_features is None or candidate_mask is None:
            return z_base, None, None
            
        delta_z, gate, res_update = self.casr(candidate_features, candidate_mask)
        z_final = z_base + res_update
        return z_final, gate, delta_z
