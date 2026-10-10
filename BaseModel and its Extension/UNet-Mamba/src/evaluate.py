import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import load_config, BASE_DIR, TEST_EVENTS
from utils import get_logger, set_seed
from model import build_model
from dataset import FloodPlanetPatchDataset
from metrics import MetricTracker
from torch.utils.data import DataLoader

def run_evaluation(config=None, checkpoint_path=None):
    if config is None:
        config = load_config()
        
    set_seed(config['experiment']['seed'])
    logger = get_logger("Evaluator")
    
    results_dir = BASE_DIR / config['evaluation']['results_dir']
    metrics_dir = BASE_DIR / config['evaluation']['metrics_dir']
    figures_dir = BASE_DIR / config['evaluation']['figures_dir']
    ckpt_dir = BASE_DIR / config['training']['save_dir']
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    if checkpoint_path is None:
        checkpoint_path = ckpt_dir / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = ckpt_dir / "last_checkpoint.pth"
            
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.eval()
    
    processed_dir = BASE_DIR / config['data']['processed_dir']
    test_csv = processed_dir / "test_patches.csv"
    df_test_patches = pd.read_csv(test_csv)
    
    norm_mean = config['data']['norm_mean']
    norm_std = config['data']['norm_std']
    
    combined_tracker = MetricTracker(num_classes=config['model']['num_classes'])
    event_trackers = {evt: MetricTracker(num_classes=config['model']['num_classes']) for evt in TEST_EVENTS}
    
    logger.info(f"Evaluating model on held-out test events: {TEST_EVENTS} ({len(df_test_patches)} patches)...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(df_test_patches)), desc="Evaluating Test Set"):
            row = df_test_patches.iloc[idx]
            event_name = row["event"]
            patch_path = row["patch_path"]
            
            data = np.load(patch_path)
            img = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.int64)
            
            # Apply channel normalization
            img_norm = (img - np.array(norm_mean, dtype=np.float32).reshape(4, 1, 1)) / (np.array(norm_std, dtype=np.float32).reshape(4, 1, 1) + 1e-6)
            
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float().to(device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).long().to(device)
            
            logits = model(img_tensor)  # (1, 2, 512, 512)
            preds = torch.argmax(logits, dim=1)
            
            combined_tracker.update(preds, mask_tensor)
            if event_name in event_trackers:
                event_trackers[event_name].update(preds, mask_tensor)
                
    # Compute Combined Metrics
    combined_metrics = combined_tracker.compute()
    logger.info("==========================================")
    logger.info(f"COMBINED TEST RESULTS across {TEST_EVENTS}:")
    logger.info(f"  Mean IoU:      {combined_metrics['mean_iou']:.4f}")
    logger.info(f"  Flood IoU:     {combined_metrics['flood_iou']:.4f}")
    logger.info(f"  Background IoU:{combined_metrics['bg_iou']:.4f}")
    logger.info(f"  Flood Dice:    {combined_metrics['flood_dice']:.4f}")
    logger.info(f"  Precision:     {combined_metrics['flood_precision']:.4f}")
    logger.info(f"  Recall:        {combined_metrics['flood_recall']:.4f}")
    logger.info(f"  Accuracy:      {combined_metrics['pixel_accuracy']:.4f}")
    logger.info("==========================================")
    
    # Per-event Metrics
    per_event_records = []
    for evt in TEST_EVENTS:
        evt_m = event_trackers[evt].compute()
        logger.info(f"Event {evt:12s} | mIoU: {evt_m['mean_iou']:.4f} | Flood IoU: {evt_m['flood_iou']:.4f} | Dice: {evt_m['flood_dice']:.4f}")
        per_event_records.append({
            "event": evt,
            "bg_iou": round(evt_m['bg_iou'], 4),
            "flood_iou": round(evt_m['flood_iou'], 4),
            "mean_iou": round(evt_m['mean_iou'], 4),
            "flood_dice": round(evt_m['flood_dice'], 4),
            "flood_precision": round(evt_m['flood_precision'], 4),
            "flood_recall": round(evt_m['flood_recall'], 4),
            "accuracy": round(evt_m['pixel_accuracy'], 4)
        })
        
    df_per_event = pd.DataFrame(per_event_records)
    df_per_event.to_csv(metrics_dir / "per_event_metrics.csv", index=False)
    
    # Save Combined Metrics
    with open(metrics_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)
        
    df_combined = pd.DataFrame([combined_metrics])
    df_combined.to_csv(metrics_dir / "test_metrics.csv", index=False)
    
    # Save Confusion Matrix Figure
    cm = np.array(combined_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Background", "Flood"], yticklabels=["Background", "Flood"],
           title="Test Confusion Matrix", ylabel="True Label", xlabel="Predicted Label")
           
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
    logger.info(f"Evaluation complete! Metrics saved to {metrics_dir}")

if __name__ == "__main__":
    run_evaluation()
