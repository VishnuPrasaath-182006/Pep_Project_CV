"""
Stage 1: Baseline Checkpoint Loading & Validation Patch Tensor Shape Verification
Tests authentic UNet-Mamba instantiation, state_dict loading, input normalisation,
and forward pass inference shapes on a real validation patch.
"""

import sys
import argparse
from pathlib import Path

# Add src to Python path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import torch
from config import load_config, BASE_DIR
from model import build_model


def verify_stage1(checkpoint_path=None, patch_path=None):
    config = load_config()
    
    # 1. Determine Checkpoint Path
    if checkpoint_path is None:
        checkpoint_path = BASE_DIR / "checkpoints" / "best_model.pth"
        if not checkpoint_path.exists():
            zip_fallback = BASE_DIR / "checkpoints" / "best_model.pth.zip"
            if zip_fallback.exists():
                import shutil
                shutil.copyfile(zip_fallback, checkpoint_path)
                
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    print("=" * 60)
    print("STAGE 1: CHECKPOINT LOADING & TENSOR SHAPE VERIFICATION")
    print("=" * 60)
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 2. Build Model & Load Checkpoint
    model = build_model(config)
    model.eval()
    
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_res = model.load_state_dict(ckpt_data["model_state_dict"])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Missing keys: {len(load_res.missing_keys)}, Unexpected keys: {len(load_res.unexpected_keys)}")
    assert len(load_res.missing_keys) == 0, f"Missing keys detected: {load_res.missing_keys}"
    assert len(load_res.unexpected_keys) == 0, f"Unexpected keys detected: {load_res.unexpected_keys}"
    print(f"Recorded checkpoint epoch: {ckpt_data.get('epoch')}")
    print(f"Recorded validation mIoU: {ckpt_data.get('best_val_miou')}")
    
    # 3. Load Validation Patch
    if patch_path is None:
        patch_path = BASE_DIR / "manifests" / "processed_data" / "processed_data" / "val" / "Bolivia_BOL_1015_p0_0_0.npz"
    patch_path = Path(patch_path)
    if not patch_path.exists():
        raise FileNotFoundError(f"Validation patch not found at: {patch_path}")
        
    print("-" * 60)
    print(f"Inspecting validation patch: {patch_path.name}")
    data = np.load(patch_path)
    raw_img = data["image"]  # (4, 512, 512)
    raw_mask = data["mask"]  # (512, 512)
    
    print(f"Raw image shape: {raw_img.shape}, dtype: {raw_img.dtype}, range: [{raw_img.min()}, {raw_img.max()}]")
    print(f"Raw mask shape: {raw_mask.shape}, dtype: {raw_mask.dtype}, unique: {np.unique(raw_mask).tolist()}")
    
    assert raw_img.shape == (4, 512, 512), f"Expected image shape (4, 512, 512), got {raw_img.shape}"
    assert raw_mask.shape == (512, 512), f"Expected mask shape (512, 512), got {raw_mask.shape}"
    
    # 4. Z-Score Normalisation
    norm_mean = np.array(config["data"]["norm_mean"], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config["data"]["norm_std"], dtype=np.float32).reshape(4, 1, 1)
    img_norm = (raw_img.astype(np.float32) - norm_mean) / (norm_std + 1e-6)
    
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float()  # (1, 4, 512, 512)
    mask_tensor = torch.from_numpy(raw_mask).unsqueeze(0).long()  # (1, 512, 512)
    
    print(f"Input tensor: shape={img_tensor.shape}, dtype={img_tensor.dtype}")
    print(f"Mask tensor: shape={mask_tensor.shape}, dtype={mask_tensor.dtype}")
    
    # 5. Forward Pass
    with torch.no_grad():
        logits = model(img_tensor)
        
    print("-" * 60)
    print("Model Forward Pass Verification:")
    print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
    assert logits.shape == (1, 2, 512, 512), f"Expected logits shape (1, 2, 512, 512), got {logits.shape}"
    assert torch.isfinite(logits).all().item(), "Non-finite values detected in logits!"
    
    probs = torch.softmax(logits, dim=1)
    flood_prob = probs[:, 1, :, :]
    preds = torch.argmax(logits, dim=1)
    
    print(f"Softmax probabilities shape: {probs.shape}, range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"Flood probability map shape: {flood_prob.shape}, range: [{flood_prob.min().item():.4f}, {flood_prob.max().item():.4f}]")
    print(f"Predicted mask shape: {preds.shape}, unique classes: {torch.unique(preds).tolist()}")
    print("=" * 60)
    print("STATUS: STAGE 1 VERIFICATION PASSED SUCCESSFULLY")
    print("=" * 60)
    return True


if __name__ == "__main__":
    verify_stage1()
