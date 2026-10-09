"""
Stage 5: Fixed Budget Training Engine for CASR (E1 & E2)
Trains only the lightweight CASR adapter head (base UNetMamba and RemoteCLIP remain frozen).
- Seed: 2026
- Max epochs: 5
- Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
- Loss: Cross-Entropy + Flood Dice over valid pixels inside candidate regions only.
- Validation checkpoint selection: Mean validation Flood IoU across Bolivia and US-Dakota (macro mIoU as tie-break).
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import load_config, BASE_DIR
from model import build_model
from casr import CAVEFloodModel
from evaluate_source_chips import compute_binary_metrics


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CASRPatchDataset(torch.utils.data.Dataset):
    def __init__(self, cached_files, experiment="E1", config=None):
        self.cached_files = cached_files
        self.experiment = experiment.upper()
        if config is None:
            config = load_config()
        self.norm_mean = np.array(config["data"]["norm_mean"], dtype=np.float32).reshape(4, 1, 1)
        self.norm_std = np.array(config["data"]["norm_std"], dtype=np.float32).reshape(4, 1, 1)

    def __len__(self):
        return len(self.cached_files)

    def __getitem__(self, idx):
        p_path = self.cached_files[idx]
        data = np.load(p_path)
        
        raw_img = data["raw_image"]                     # (4, 512, 512)
        raw_mask = data["raw_mask"]                     # (512, 512)
        cand_mask = data["candidate_mask"]             # (512, 512)
        flood_prob = data["flood_prob"]                 # (512, 512)
        entropy_map = data["entropy_map"]               # (512, 512)
        grad_map = data["gradient_map"]                 # (512, 512)
        
        guidance = data["guidance_e1"] if self.experiment == "E1" else data["guidance_e2"]
        
        # Normalized imagery
        img_norm = (raw_img.astype(np.float32) - self.norm_mean) / (self.norm_std + 1e-6)
        
        # Assemble 9-channel feature tensor
        features = np.concatenate([
            img_norm,                                   # 4 channels
            flood_prob[np.newaxis, ...],                # 1 channel
            entropy_map[np.newaxis, ...],               # 1 channel
            grad_map[np.newaxis, ...],                  # 1 channel
            cand_mask[np.newaxis, ...],                 # 1 channel
            guidance[np.newaxis, ...]                   # 1 channel
        ], axis=0).astype(np.float32)
        
        return {
            "img_norm": torch.from_numpy(img_norm).float(),
            "features": torch.from_numpy(features).float(),
            "cand_mask": torch.from_numpy(cand_mask).unsqueeze(0).float(),
            "raw_img": torch.from_numpy(raw_img).long(),
            "target_mask": torch.from_numpy(raw_mask).long()
        }


def compute_casr_loss(z_final, target_mask, cand_mask, raw_img, smooth=1e-5):
    """
    Computes cross-entropy plus flood Dice over valid pixels inside candidate regions only.
    """
    # Valid pixel definition: not all 4 bands are 0 and label is not -1
    img_valid = (raw_img != 0).any(dim=1)               # (B, H, W)
    label_valid = (target_mask != -1)                   # (B, H, W)
    in_cand = (cand_mask.squeeze(1) == 1.0)             # (B, H, W)
    
    loss_mask = img_valid & label_valid & in_cand       # (B, H, W)
    
    if loss_mask.sum() == 0:
        return torch.tensor(0.0, device=z_final.device, requires_grad=True)
        
    logits_flat = z_final.permute(0, 2, 3, 1)[loss_mask]    # (N, 2)
    targets_flat = target_mask[loss_mask]                   # (N,)
    
    # 1. Cross-Entropy Loss
    ce_loss = F.cross_entropy(logits_flat, targets_flat)
    
    # 2. Flood Dice Loss
    probs = torch.softmax(z_final, dim=1)[:, 1]             # (B, H, W)
    p_flood = probs[loss_mask]                              # (N,)
    y_flood = (targets_flat == 1).float()                   # (N,)
    
    intersection = (p_flood * y_flood).sum()
    cardinality = p_flood.sum() + y_flood.sum()
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)
    
    total_loss = ce_loss + dice_loss
    return total_loss, ce_loss.item(), dice_loss.item()


def evaluate_val_chips_casr(model, val_cached_files, experiment="E1", device="cpu"):
    """
    Fast validation evaluation: reconstructs chips using cached features and CASR predictions.
    """
    model.eval()
    
    # Group files by scene (event_stem)
    scenes = {}
    for f in val_cached_files:
        stem_key = f.stem.rsplit("_p", 1)[0]
        p_idx = int(f.stem.rsplit("_p", 1)[1])
        if stem_key not in scenes:
            scenes[stem_key] = {}
        scenes[stem_key][p_idx] = f
        
    quadrant_coords = {
        0: (0, 0),
        1: (0, 512),
        2: (512, 0),
        3: (512, 512)
    }
    
    event_counts = {"Bolivia": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "chips": 0},
                    "US-Dakota": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "chips": 0}}
    pooled_tp, pooled_fp, pooled_fn, pooled_tn = 0, 0, 0, 0
    
    for stem_key, p_map in scenes.items():
        event_name = stem_key.split("_", 1)[0]
        if event_name not in event_counts:
            event_counts[event_name] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "chips": 0}
            
        recon_logits = np.zeros((2, 1024, 1024), dtype=np.float32)
        recon_img = np.zeros((4, 1024, 1024), dtype=np.uint16)
        recon_mask = np.full((1024, 1024), -1, dtype=np.int64)
        
        for p_idx in range(4):
            if p_idx not in p_map:
                continue
            top, left = quadrant_coords[p_idx]
            data = np.load(p_map[p_idx])
            
            recon_img[:, top:top+512, left:left+512] = data["raw_image"]
            recon_mask[top:top+512, left:left+512] = data["raw_mask"]
            
            # Fast CASR forward
            z_base = torch.from_numpy(data["patch_logits"].astype(np.float32)).unsqueeze(0).to(device)
            cand_mask = torch.from_numpy(data["candidate_mask"]).unsqueeze(0).unsqueeze(0).to(device)
            
            guidance = data["guidance_e1"] if experiment == "E1" else data["guidance_e2"]
            norm_mean = np.array([2334.81, 2378.43, 2185.12, 3608.97], dtype=np.float32).reshape(4, 1, 1)
            norm_std = np.array([1245.52, 1184.21, 1148.65, 1420.33], dtype=np.float32).reshape(4, 1, 1)
            img_norm = (data["raw_image"].astype(np.float32) - norm_mean) / (norm_std + 1e-6)
            
            feat = np.concatenate([
                img_norm,
                data["flood_prob"][np.newaxis, ...],
                data["entropy_map"][np.newaxis, ...],
                data["gradient_map"][np.newaxis, ...],
                data["candidate_mask"][np.newaxis, ...],
                guidance[np.newaxis, ...]
            ], axis=0).astype(np.float32)
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
            
            with torch.no_grad():
                delta_z, gate, res_update = model.casr(feat_t, cand_mask)
                z_final = z_base + res_update
                
            recon_logits[:, top:top+512, left:left+512] = z_final[0].cpu().numpy()
            
        # Validity mask
        img_valid = (recon_img != 0).any(axis=0)
        mask_valid = (recon_mask != -1)
        valid_pixels = img_valid & mask_valid
        
        preds = np.argmax(recon_logits, axis=0)
        y_true = recon_mask[valid_pixels]
        y_pred = preds[valid_pixels]
        
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        
        pooled_tp += tp
        pooled_fp += fp
        pooled_fn += fn
        pooled_tn += tn
        
        event_counts[event_name]["tp"] += tp
        event_counts[event_name]["fp"] += fp
        event_counts[event_name]["fn"] += fn
        event_counts[event_name]["tn"] += tn
        event_counts[event_name]["chips"] += 1
        
    per_event_res = {}
    for evt, cnt in event_counts.items():
        if cnt["chips"] > 0:
            per_event_res[evt] = compute_binary_metrics(cnt["tp"], cnt["fp"], cnt["fn"], cnt["tn"], cnt["chips"])
            
    pooled_res = compute_binary_metrics(pooled_tp, pooled_fp, pooled_fn, pooled_tn, len(scenes))
    
    # Event mean flood IoU (Bolivia + US-Dakota) / 2
    evt_flood_ious = [per_event_res[e]["flood_iou"] for e in ["Bolivia", "US-Dakota"] if e in per_event_res]
    mean_evt_flood_iou = float(np.mean(evt_flood_ious)) if evt_flood_ious else 0.0
    
    evt_mean_ious = [per_event_res[e]["mean_iou"] for e in ["Bolivia", "US-Dakota"] if e in per_event_res]
    mean_evt_miou = float(np.mean(evt_mean_ious)) if evt_mean_ious else 0.0
    
    return {
        "pooled": pooled_res,
        "per_event": per_event_res,
        "mean_event_flood_iou": mean_evt_flood_iou,
        "mean_event_miou": mean_evt_miou
    }


def run_training_casr(
    experiment="E1",
    epochs=5,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=2,
    seed=2026
):
    experiment = experiment.upper()
    assert experiment in ["E1", "E2"], f"Invalid experiment: {experiment}"
    assert epochs <= 5, f"Maximum 5 epochs allowed by protocol! Got {epochs}"
    
    set_seed(seed)
    device = torch.device("cpu")
    config = load_config()
    
    print("\n" + "=" * 70)
    print(f"STAGE 5: CASR TRAINING RUN [{experiment}] (SEED {seed})")
    print(f"Mode: {'E1 (Neutral VLM Guidance)' if experiment == 'E1' else 'E2 (RemoteCLIP Guidance)'}")
    print(f"Hyperparameters: Epochs={epochs}, LR={lr}, WeightDecay={weight_decay}, BatchSize={batch_size}")
    print("=" * 70)
    
    # 1. Build Base UNetMamba and CAVE-Flood Wrapper
    base_model = build_model(config).to(device)
    base_model.eval()
    ckpt_path = BASE_DIR / "checkpoints" / "best_model.pth"
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt_data["model_state_dict"])
    
    cave_model = CAVEFloodModel(base_model, in_channels=9, hidden_channels=32, lambda_res=1.0).to(device)
    
    # 2. Setup Optimizer (Train only CASR adapter)
    optimizer = torch.optim.AdamW(cave_model.casr.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 3. Load Datasets
    train_cache_dir = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "cache" / "candidate_regions" / "train"
    val_cache_dir = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "cache" / "candidate_regions" / "val"
    
    train_files = sorted(list(train_cache_dir.glob("*.npz")))
    val_files = sorted(list(val_cache_dir.glob("*.npz")))
    
    assert len(train_files) > 0, "No cached training patches found! Run cache_train_candidate_regions.py first!"
    assert len(val_files) > 0, "No cached validation patches found!"
    
    train_dataset = CASRPatchDataset(train_files, experiment=experiment, config=config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    print(f"Training dataset size: {len(train_dataset)} patches across 14 events")
    print(f"Validation dataset size: {len(val_files)} patches across 114 scenes (Bolivia, US-Dakota)")
    
    # 4. Initial E0 Validation Score as Baseline
    print("\nComputing Initial Zero-Residual Validation Score...")
    val_e0 = evaluate_val_chips_casr(cave_model, val_files, experiment=experiment, device=device)
    best_mean_flood_iou = val_e0["mean_event_flood_iou"]
    best_mean_miou = val_e0["mean_event_miou"]
    best_epoch = 0
    
    print(f"Epoch 0 (Baseline E0 Equivalence):")
    print(f"  Mean Event Flood IoU: {best_mean_flood_iou:.4f} (Bolivia: {val_e0['per_event']['Bolivia']['flood_iou']:.4f}, US-Dakota: {val_e0['per_event']['US-Dakota']['flood_iou']:.4f})")
    print(f"  Pixel-Pooled Flood IoU: {val_e0['pooled']['flood_iou']:.4f}")
    
    ckpt_dir = BASE_DIR / "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    logs_dir = BASE_DIR / "results" / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save Initial State as Best Baseline
    best_ckpt_path = ckpt_dir / f"best_{experiment.lower()}_model.pth"
    torch.save({
        "epoch": 0,
        "experiment": experiment,
        "casr_state_dict": cave_model.casr.state_dict(),
        "mean_event_flood_iou": best_mean_flood_iou,
        "mean_event_miou": best_mean_miou,
        "seed": seed
    }, best_ckpt_path)
    
    history = []
    
    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        cave_model.casr.train()
        epoch_losses = []
        epoch_ce = []
        epoch_dice = []
        
        start_t = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            
            features = batch["features"].to(device)
            cand_mask = batch["cand_mask"].to(device)
            raw_img = batch["raw_img"].to(device)
            target_mask = batch["target_mask"].to(device)
            
            # CASR forward: pass 9-channel features
            delta_z, gate, res_update = cave_model.casr(features, cand_mask)
            
            # Baseline logits from feature channel 1 (flood prob)
            z_base = features[:, 1:3, :, :]  # Use delta_z directly or z_base + res_update
            # Full z_final = z_base + lambda * C * G * delta_z
            z_final = delta_z  # Gradient flows through delta_z and gate directly to targets
            
            loss, ce_val, dice_val = compute_casr_loss(
                res_update, target_mask, cand_mask, raw_img
            )
            
            if torch.isfinite(loss) and loss.requires_grad:
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                epoch_ce.append(ce_val)
                epoch_dice.append(dice_val)
                
        train_time = time.time() - start_t
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        
        # Validate each epoch on Bolivia and US-Dakota
        val_res = evaluate_val_chips_casr(cave_model, val_files, experiment=experiment, device=device)
        cur_mean_flood_iou = val_res["mean_event_flood_iou"]
        cur_mean_miou = val_res["mean_event_miou"]
        
        # Checkpoint selection rule: mean validation flood IoU across events, tie-break on macro mIoU
        improved = (cur_mean_flood_iou > best_mean_flood_iou) or (
            cur_mean_flood_iou == best_mean_flood_iou and cur_mean_miou > best_mean_miou
        )
        
        if improved:
            best_mean_flood_iou = cur_mean_flood_iou
            best_mean_miou = cur_mean_miou
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "experiment": experiment,
                "casr_state_dict": cave_model.casr.state_dict(),
                "mean_event_flood_iou": best_mean_flood_iou,
                "mean_event_miou": best_mean_miou,
                "seed": seed
            }, best_ckpt_path)
            star = " [* BEST]"
        else:
            star = ""
            
        epoch_rec = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_mean_event_flood_iou": cur_mean_flood_iou,
            "val_mean_event_miou": cur_mean_miou,
            "val_bolivia_flood_iou": val_res["per_event"]["Bolivia"]["flood_iou"],
            "val_usdakota_flood_iou": val_res["per_event"]["US-Dakota"]["flood_iou"],
            "val_pooled_flood_iou": val_res["pooled"]["flood_iou"],
            "val_pooled_dice": val_res["pooled"]["flood_dice"],
            "is_best": improved
        }
        history.append(epoch_rec)
        
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Mean Event Flood IoU: {cur_mean_flood_iou:.4f} | Bolivia: {val_res['per_event']['Bolivia']['flood_iou']:.4f} | US-Dakota: {val_res['per_event']['US-Dakota']['flood_iou']:.4f}{star}")
        
    # Save Final Checkpoint
    final_ckpt_path = ckpt_dir / f"final_{experiment.lower()}_epoch_{epochs}.pth"
    torch.save({
        "epoch": epochs,
        "experiment": experiment,
        "casr_state_dict": cave_model.casr.state_dict(),
        "seed": seed
    }, final_ckpt_path)
    
    # Save History CSV & JSON
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(logs_dir / f"{experiment.lower()}_training_history.csv", index=False)
    with open(logs_dir / f"{experiment.lower()}_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print("-" * 70)
    print(f"[{experiment} TRAINING COMPLETE]")
    print(f"Best Validation Epoch: {best_epoch} with Mean Event Flood IoU: {best_mean_flood_iou:.4f}")
    print(f"Saved Best Checkpoint to: {best_ckpt_path}")
    print(f"Saved Training History to: {logs_dir / f'{experiment.lower()}_training_history.csv'}")
    print("=" * 70)
    return best_epoch, best_mean_flood_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="E1", choices=["E1", "E2"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    
    run_training_casr(experiment=args.experiment, epochs=args.epochs, lr=args.lr, seed=args.seed)
