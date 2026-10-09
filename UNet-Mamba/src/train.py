import os
import sys
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import load_config, BASE_DIR
from utils import set_seed, get_logger, get_gpu_memory_mb
from model import build_model
from losses import build_loss
from dataset import get_dataloaders
from metrics import MetricTracker

def run_training(config=None, resume=True):
    if config is None:
        config = load_config()
        
    set_seed(config['experiment']['seed'])
    
    logs_dir = BASE_DIR / config['training']['logs_dir']
    ckpt_dir = BASE_DIR / config['training']['save_dir']
    figures_dir = BASE_DIR / config['evaluation']['figures_dir']
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    logger = get_logger("TrainEngine", log_file=logs_dir / "train.log")
    logger.info("Initializing UNet-Mamba Training Engine...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Build Model
    model = build_model(config).to(device)
    
    # 2. Separate Optimizer Parameter Groups (Backbone vs Non-Backbone)
    backbone_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['training']['backbone_learning_rate']},
        {'params': decoder_params, 'lr': config['training']['learning_rate']}
    ], weight_decay=config['training']['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['scheduler_t0'],
        T_mult=config['training']['scheduler_t_mult']
    )
    
    criterion = build_loss(config)
    
    # 3. Load Dataloaders
    train_loader, val_loader, _ = get_dataloaders(config)
    
    start_epoch = 1
    total_epochs = config['training']['epochs']  # 21
    best_val_miou = 0.0
    history = []
    
    # 4. Resume from Checkpoint if exists
    last_ckpt_path = ckpt_dir / "last_checkpoint.pth"
    if resume and last_ckpt_path.exists():
        logger.info(f"Loading checkpoint from: {last_ckpt_path}")
        checkpoint_data = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        start_epoch = checkpoint_data['epoch'] + 1
        best_val_miou = checkpoint_data.get('best_val_miou', 0.0)
        history = checkpoint_data.get('history', [])
        logger.info(f"Resumed training successfully from Epoch {start_epoch}")
        
    if start_epoch > total_epochs:
        logger.info(f"Training already completed ({start_epoch - 1}/{total_epochs} epochs). Skipping training.")
        return
        
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    logger.info(f"AMP Precision Mode: {amp_dtype}")
    
    # 5. Training Loop
    logger.info(f"Starting {total_epochs}-Epoch Training Loop (Epoch {start_epoch} to {total_epochs})...")
    
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()
        
        # --- TRAIN EPOCH ---
        model.train()
        train_loss_sum = 0.0
        train_tracker = MetricTracker(num_classes=config['model']['num_classes'])
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{total_epochs} [Train]")
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected at Epoch {epoch}: loss = {loss.item()}")
                
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss_sum += loss.item()
            main_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            train_tracker.update(main_logits, masks)
            
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        train_loss = train_loss_sum / len(train_loader)
        train_metrics = train_tracker.compute()
        
        # --- VALIDATION EPOCH ---
        model.eval()
        val_loss_sum = 0.0
        val_tracker = MetricTracker(num_classes=config['model']['num_classes'])
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{total_epochs} [Val]")
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                val_loss_sum += loss.item()
                val_tracker.update(outputs, masks)
                val_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
        val_loss = val_loss_sum / len(val_loader)
        val_metrics = val_tracker.compute()
        
        # Step LR Scheduler
        scheduler.step()
        
        epoch_duration = time.time() - epoch_start_time
        peak_gpu_mem_mb = get_gpu_memory_mb()
        
        lr_backbone = optimizer.param_groups[0]['lr']
        lr_decoder = optimizer.param_groups[1]['lr']
        
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_bg_iou": round(val_metrics['bg_iou'], 4),
            "val_flood_iou": round(val_metrics['flood_iou'], 4),
            "val_mean_iou": round(val_metrics['mean_iou'], 4),
            "val_dice": round(val_metrics['flood_dice'], 4),
            "val_precision": round(val_metrics['flood_precision'], 4),
            "val_recall": round(val_metrics['flood_recall'], 4),
            "val_accuracy": round(val_metrics['pixel_accuracy'], 4),
            "lr_backbone": lr_backbone,
            "lr_decoder": lr_decoder,
            "peak_gpu_mem_mb": round(peak_gpu_mem_mb, 1),
            "epoch_duration_sec": round(epoch_duration, 2)
        }
        history.append(log_entry)
        
        logger.info(
            f"Epoch {epoch:02d}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_metrics['mean_iou']:.4f} | "
            f"Flood IoU: {val_metrics['flood_iou']:.4f} | "
            f"Flood Dice: {val_metrics['flood_dice']:.4f} | "
            f"Time: {epoch_duration:.1f}s | "
            f"VRAM: {peak_gpu_mem_mb:.1f}MB"
        )
        
        # Save Last Checkpoint
        ckpt_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_miou': best_val_miou,
            'history': history,
            'config': config
        }
        torch.save(ckpt_state, last_ckpt_path)
        
        # Save Best Checkpoint based on Validation Mean IoU
        if val_metrics['mean_iou'] > best_val_miou:
            best_val_miou = val_metrics['mean_iou']
            ckpt_state['best_val_miou'] = best_val_miou
            torch.save(ckpt_state, ckpt_dir / "best_model.pth")
            logger.info(f"Saved new BEST model checkpoint with Val mIoU: {best_val_miou:.4f}")
            
    # Save Final Epoch 21 Checkpoint
    final_ckpt_path = ckpt_dir / f"final_epoch_{total_epochs}.pth"
    torch.save(ckpt_state, final_ckpt_path)
    logger.info(f"Saved final epoch checkpoint to {final_ckpt_path}")
    
    # Save Training History CSV & JSON
    df_history = pd.DataFrame(history)
    df_history.to_csv(logs_dir / "training_history.csv", index=False)
    with open(logs_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    # Generate Training Curves Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs_arr = [h['epoch'] for h in history]
    
    axes[0, 0].plot(epochs_arr, [h['train_loss'] for h in history], label="Train Loss", color="blue")
    axes[0, 0].plot(epochs_arr, [h['val_loss'] for h in history], label="Val Loss", color="red")
    axes[0, 0].set_title("Train & Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs_arr, [h['val_mean_iou'] for h in history], label="Val mIoU", color="green")
    axes[0, 1].plot(epochs_arr, [h['val_flood_iou'] for h in history], label="Val Flood IoU", color="purple")
    axes[0, 1].set_title("Validation IoU Metrics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("IoU Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs_arr, [h['val_dice'] for h in history], label="Val Flood Dice/F1", color="orange")
    axes[1, 0].set_title("Validation Flood Dice Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Dice Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs_arr, [h['lr_decoder'] for h in history], label="Decoder LR", color="magenta")
    axes[1, 1].plot(epochs_arr, [h['lr_backbone'] for h in history], label="Backbone LR", color="cyan")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "training_curves.png", dpi=150)
    plt.close()
    
    logger.info(f"Training completed successfully for {total_epochs} total epochs!")

if __name__ == "__main__":
    run_training()
