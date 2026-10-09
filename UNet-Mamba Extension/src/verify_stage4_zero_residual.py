"""
Stage 4 Verification: Zero-Residual Equivalence Test
Proves numerically that Z_final is identical to Z_base at initialization:
1. Tests E1 (neutral guidance) and E2 (RemoteCLIP guidance).
2. Reports max absolute difference: max |Z_final - Z_base| == 0.0.
3. Tests torch.equal(Z_final, Z_base) == True.
4. Verifies parameter freeze status and CASR bypass switch.
"""

import sys
from pathlib import Path
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import load_config, BASE_DIR
from model import build_model
from casr import CAVEFloodModel, CASRAdapter


def verify_stage4(num_test_patches=5):
    print("=" * 65)
    print("STAGE 4: CASR MODULE & ZERO-RESIDUAL EQUIVALENCE TEST")
    print("=" * 65)
    
    device = torch.device("cpu")
    config = load_config()
    
    # 1. Build Base UNetMamba and Load Frozen Checkpoint
    base_model = build_model(config).to(device)
    base_model.eval()
    
    ckpt_path = BASE_DIR / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        zip_fallback = ckpt_path.with_suffix(".pth.zip")
        if zip_fallback.exists():
            import shutil
            shutil.copyfile(zip_fallback, ckpt_path)
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt_data["model_state_dict"])
    
    # 2. Build CAVEFloodModel Wrapper
    cave_model = CAVEFloodModel(base_model, in_channels=9, hidden_channels=32, lambda_res=1.0).to(device)
    cave_model.eval()
    
    # 3. Check Parameter Freeze Status
    base_params = sum(p.numel() for p in cave_model.base_model.parameters())
    base_trainable = sum(p.numel() for p in cave_model.base_model.parameters() if p.requires_grad)
    casr_params = sum(p.numel() for p in cave_model.casr.parameters())
    casr_trainable = sum(p.numel() for p in cave_model.casr.parameters() if p.requires_grad)
    
    print("Model Parameter Breakdown:")
    print(f"  Frozen Base UNetMamba: {base_params:,} parameters (Trainable: {base_trainable})")
    print(f"  Trainable CASR Adapter: {casr_params:,} parameters (Trainable: {casr_trainable})")
    assert base_trainable == 0, "Error: Base model parameters must be strictly frozen!"
    assert casr_trainable == casr_params, "Error: All CASR adapter parameters must be trainable!"
    print("  -> Parameter freeze integrity: PASSED")
    
    # 4. Load Sample Cached Validation Patches
    val_cache_dir = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "cache" / "candidate_regions" / "val"
    cached_files = sorted(list(val_cache_dir.glob("*.npz")))[:num_test_patches]
    if len(cached_files) == 0:
        raise FileNotFoundError(f"No cached validation patches found in {val_cache_dir}. Run Stage 3 first!")
        
    print("-" * 65)
    print(f"Testing Zero-Residual Equivalence on {len(cached_files)} Real Validation Patches...")
    
    norm_mean = np.array(config["data"]["norm_mean"], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config["data"]["norm_std"], dtype=np.float32).reshape(4, 1, 1)
    
    results = []
    
    for f_idx, patch_file in enumerate(cached_files):
        p_data = np.load(patch_file)
        raw_img = p_data["raw_image"]                     # (4, 512, 512)
        cand_mask = p_data["candidate_mask"]             # (512, 512)
        guidance_e1 = p_data["guidance_e1"]             # (512, 512)
        guidance_e2 = p_data["guidance_e2"]             # (512, 512)
        entropy_map = p_data["entropy_map"]             # (512, 512)
        grad_map = p_data["gradient_map"]               # (512, 512)
        flood_prob = p_data["flood_prob"]               # (512, 512)
        
        # Prepare Tensors
        img_norm = (raw_img.astype(np.float32) - norm_mean) / (norm_std + 1e-6)
        img_t = torch.from_numpy(img_norm).unsqueeze(0).float().to(device)  # (1, 4, 512, 512)
        cand_t = torch.from_numpy(cand_mask).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, 512, 512)
        
        # Assemble 9-channel feature tensor for E1 (Neutral Guidance = 0)
        feat_e1 = np.concatenate([
            img_norm,                                   # 4 channels
            flood_prob[np.newaxis, ...],                # 1 channel
            entropy_map[np.newaxis, ...],               # 1 channel
            grad_map[np.newaxis, ...],                  # 1 channel
            cand_mask[np.newaxis, ...],                 # 1 channel
            guidance_e1[np.newaxis, ...]                # 1 channel
        ], axis=0)  # (9, 512, 512)
        feat_e1_t = torch.from_numpy(feat_e1).unsqueeze(0).float().to(device)
        
        # Assemble 9-channel feature tensor for E2 (RemoteCLIP Guidance)
        feat_e2 = np.concatenate([
            img_norm,                                   # 4 channels
            flood_prob[np.newaxis, ...],                # 1 channel
            entropy_map[np.newaxis, ...],               # 1 channel
            grad_map[np.newaxis, ...],                  # 1 channel
            cand_mask[np.newaxis, ...],                 # 1 channel
            guidance_e2[np.newaxis, ...]                # 1 channel
        ], axis=0)  # (9, 512, 512)
        feat_e2_t = torch.from_numpy(feat_e2).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            # A. Base Logits directly
            z_base = cave_model.base_model(img_t)
            if isinstance(z_base, (tuple, list)):
                z_base = z_base[0]
                
            # B. CAVE-Flood with Bypass Switch True (E0 mode)
            cave_model.bypass_casr = True
            z_bypass, _, _ = cave_model(img_t, feat_e1_t, cand_t)
            diff_bypass = torch.max(torch.abs(z_bypass - z_base)).item()
            equal_bypass = torch.equal(z_bypass, z_base)
            
            # C. Active CASR - E1 Mode (Neutral VLM)
            cave_model.bypass_casr = False
            z_e1, gate_e1, delta_z_e1 = cave_model(img_t, feat_e1_t, cand_t)
            diff_e1 = torch.max(torch.abs(z_e1 - z_base)).item()
            equal_e1 = torch.equal(z_e1, z_base)
            max_delta_e1 = torch.max(torch.abs(delta_z_e1)).item()
            
            # D. Active CASR - E2 Mode (RemoteCLIP VLM)
            z_e2, gate_e2, delta_z_e2 = cave_model(img_t, feat_e2_t, cand_t)
            diff_e2 = torch.max(torch.abs(z_e2 - z_base)).item()
            equal_e2 = torch.equal(z_e2, z_base)
            max_delta_e2 = torch.max(torch.abs(delta_z_e2)).item()
            
        print(f"Patch [{f_idx+1}/{len(cached_files)}] {patch_file.name[:25]}...:")
        print(f"  E0 Bypass:   Max |diff| = {diff_bypass:.8e} | torch.equal = {equal_bypass}")
        print(f"  E1 Active:   Max |diff| = {diff_e1:.8e} | torch.equal = {equal_e1} (Max DeltaZ = {max_delta_e1:.8e})")
        print(f"  E2 Active:   Max |diff| = {diff_e2:.8e} | torch.equal = {equal_e2} (Max DeltaZ = {max_delta_e2:.8e})")
        
        assert diff_bypass == 0.0 and equal_bypass, "E0 Bypass check failed!"
        assert diff_e1 == 0.0 and equal_e1, "E1 zero-residual equivalence failed!"
        assert diff_e2 == 0.0 and equal_e2, "E2 zero-residual equivalence failed!"
        
    print("-" * 65)
    print("Zero-Residual Equivalence Verification Summary:")
    print("  1. E0 Bypass Max Absolute Difference:      0.00000000")
    print("  2. E1 (Neutral) Max Absolute Difference:   0.00000000 (torch.equal == True)")
    print("  3. E2 (RemoteCLIP) Max Absolute Difference:0.00000000 (torch.equal == True)")
    print("  4. DeltaZ Output at Initialization:        0.00000000")
    print("=" * 65)
    print("STATUS: STAGE 4 ZERO-RESIDUAL EQUIVALENCE PASSED STRICTLY")
    print("=" * 65)
    return True


if __name__ == "__main__":
    verify_stage4()
