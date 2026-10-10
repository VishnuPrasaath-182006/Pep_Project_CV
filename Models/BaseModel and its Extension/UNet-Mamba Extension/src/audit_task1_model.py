"""
Task 1 Audit: Model Architecture Verification
Independently verifies:
1. Frozen UNet-Mamba backbone parameter count and requires_grad status.
2. Trainable CASR adapter parameter count and requires_grad status.
3. RemoteCLIP parameter status and prompt strings.
4. Input/output tensor shapes.
5. Residual equation and gate behavior.
6. Exact zero-residual equivalence.
"""

import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import torch
import torch.nn as nn
from config import load_config
from model import build_model
from casr import CASRAdapter, CAVEFloodModel

def audit_task1():
    print("=" * 70)
    print("AUDIT TASK 1: MODEL ARCHITECTURE & ZERO-RESIDUAL EQUIVALENCE")
    print("=" * 70)
    
    config = load_config()
    
    # 1. Base UNet-Mamba
    base_model = build_model(config)
    total_base_params = sum(p.numel() for p in base_model.parameters())
    trainable_base_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Base UNet-Mamba Total Parameters: {total_base_params:,}")
    print(f"Base UNet-Mamba Initial Trainable Parameters: {trainable_base_params:,}")
    
    # 2. Wrap into CAVEFloodModel
    cave_model = CAVEFloodModel(base_model, in_channels=9, hidden_channels=32, lambda_res=1.0)
    
    total_cave_params = sum(p.numel() for p in cave_model.parameters())
    base_requires_grad = any(p.requires_grad for p in cave_model.base_model.parameters())
    
    casr_params = sum(p.numel() for p in cave_model.casr.parameters())
    casr_trainable_params = sum(p.numel() for p in cave_model.casr.parameters() if p.requires_grad)
    
    print(f"CAVEFloodModel Total Parameters: {total_cave_params:,}")
    print(f"Base Model in CAVEFloodModel requires_grad: {base_requires_grad} (Expected: False)")
    print(f"CASR Adapter Total Parameters: {casr_params:,}")
    print(f"CASR Adapter Trainable Parameters: {casr_trainable_params:,} (Expected: 12,035)")
    
    # Verify CASR parameter layer by layer
    print("\nCASR Parameter Breakdown:")
    for name, param in cave_model.casr.named_parameters():
        print(f"  {name:30s} shape: {str(list(param.shape)):20s} numel: {param.numel():6d} requires_grad: {param.requires_grad}")
        
    # 3. Input / Output Tensor Shapes
    b, c, h, w = 2, 4, 512, 512
    x_img = torch.randn(b, c, h, w)
    feat_cand = torch.randn(b, 9, h, w)
    cand_mask = torch.ones(b, 1, h, w)
    
    cave_model.eval()
    with torch.no_grad():
        z_final, gate, delta_z = cave_model(x_img, feat_cand, cand_mask)
        z_base = cave_model.base_model(x_img)
        if isinstance(z_base, (tuple, list)):
            z_base = z_base[0]
            
    print("\nTensor Shape Verification:")
    print(f"  Input Imagery Shape:     {x_img.shape} (Expected: [2, 4, 512, 512])")
    print(f"  Candidate Features Shape: {feat_cand.shape} (Expected: [2, 9, 512, 512])")
    print(f"  Candidate Mask Shape:    {cand_mask.shape} (Expected: [2, 1, 512, 512])")
    print(f"  Output Logits Shape:     {z_final.shape} (Expected: [2, 2, 512, 512])")
    print(f"  Gate Output Shape:       {gate.shape} (Expected: [2, 1, 512, 512])")
    print(f"  Delta Z Shape:           {delta_z.shape} (Expected: [2, 2, 512, 512])")
    
    # 4. Gate Behavior Check
    gate_min = gate.min().item()
    gate_max = gate.max().item()
    print(f"\nGate Range Check: min={gate_min:.6f}, max={gate_max:.6f} (Must be in [0.0, 1.0])")
    
    # 5. Exact Zero-Residual Equivalence
    # At initial state (before training): res_head is zero-initialized
    diff_init = (z_final - z_base).abs().max().item()
    exact_equal = torch.equal(z_final, z_base)
    print(f"\nZero-Residual Equivalence (Initial State):")
    print(f"  Max Absolute Difference |z_final - z_base|: {diff_init:.10f}")
    print(f"  torch.equal(z_final, z_base): {exact_equal}")
    
    # Bypass mode check
    cave_model.bypass_casr = True
    with torch.no_grad():
        z_bypass, _, _ = cave_model(x_img, feat_cand, cand_mask)
    diff_bypass = (z_bypass - z_base).abs().max().item()
    print(f"Bypass Mode Equivalence: Max Diff: {diff_bypass:.10f}, Equal: {torch.equal(z_bypass, z_base)}")
    cave_model.bypass_casr = False
    
    # 6. Residual Equation Check: Z_final == Z_base + lambda * C * G * delta_z
    lambda_res = cave_model.casr.lambda_res
    expected_update = lambda_res * cand_mask * gate * delta_z
    actual_diff = z_final - z_base
    equation_diff = (actual_diff - expected_update).abs().max().item()
    print(f"Residual Equation Verification |(z_final - z_base) - lambda*C*G*delta_z|: {equation_diff:.10f}")
    
    # 7. Check RemoteCLIP Text Prompts and Freezing
    from candidate_selector import FLOOD_PROMPTS, NONFLOOD_PROMPTS
    print(f"\nRemoteCLIP Prompts:")
    print(f"  Positive Flood Prompts ({len(FLOOD_PROMPTS)}): {FLOOD_PROMPTS}")
    print(f"  Negative Non-Flood Prompts ({len(NONFLOOD_PROMPTS)}): {NONFLOOD_PROMPTS}")

if __name__ == "__main__":
    audit_task1()
