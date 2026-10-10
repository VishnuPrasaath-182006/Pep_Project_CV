import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from config import load_config, BASE_DIR, TEST_EVENTS
from utils import get_logger
from model import build_model

def generate_visualizations(config=None, checkpoint_path=None, num_samples=6):
    if config is None:
        config = load_config()
    logger = get_logger("Visualizer")
    
    figures_dir = BASE_DIR / config['evaluation']['figures_dir'] / "test_comparisons"
    ckpt_dir = BASE_DIR / config['training']['save_dir']
    os.makedirs(figures_dir, exist_ok=True)
    
    if checkpoint_path is None:
        checkpoint_path = ckpt_dir / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = ckpt_dir / "last_checkpoint.pth"
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.eval()
    
    processed_dir = BASE_DIR / config['data']['processed_dir']
    test_csv = processed_dir / "test_patches.csv"
    df_test = pd.read_csv(test_csv)
    
    norm_mean = config['data']['norm_mean']
    norm_std = config['data']['norm_std']
    
    sample_indices = np.linspace(0, len(df_test)-1, min(num_samples, len(df_test)), dtype=int)
    
    logger.info(f"Generating segmentation visualizations for {len(sample_indices)} test patches...")
    
    with torch.no_grad():
        for s_idx in sample_indices:
            row = df_test.iloc[s_idx]
            event_name = row["event"]
            stem = row["stem"]
            patch_idx = row["patch_idx"]
            
            data = np.load(row["patch_path"])
            img = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.int64)
            
            img_norm = (img - np.array(norm_mean, dtype=np.float32).reshape(4, 1, 1)) / (np.array(norm_std, dtype=np.float32).reshape(4, 1, 1) + 1e-6)
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float().to(device)
            
            logits = model(img_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Error Map construction
            # TP = 1 & pred = 1 (Blue/Green)
            # TN = 0 & pred = 0 (Black/Grey)
            # FP = 0 & pred = 1 (Red)
            # FN = 1 & pred = 0 (Yellow)
            error_map = np.zeros((*mask.shape, 3), dtype=np.float32)
            
            valid = (mask != -1)
            tp_m = valid & (mask == 1) & (pred == 1)
            tn_m = valid & (mask == 0) & (pred == 0)
            fp_m = valid & (mask == 0) & (pred == 1)
            fn_m = valid & (mask == 1) & (pred == 0)
            
            error_map[tp_m] = [0.0, 0.8, 0.2]  # Green for TP
            error_map[tn_m] = [0.2, 0.2, 0.2]  # Dark Grey for TN
            error_map[fp_m] = [0.9, 0.1, 0.1]  # Red for False Positive
            error_map[fn_m] = [0.9, 0.8, 0.1]  # Yellow for False Negative
            
            # RGB visualization
            rgb = img[:3, :, :].transpose(1, 2, 0).astype(np.float32)
            rgb_norm = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
            
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(rgb_norm)
            axes[0].set_title(f"PlanetScope RGB ({event_name})")
            axes[0].axis("off")
            
            axes[1].imshow(mask, cmap="coolwarm", vmin=-1, vmax=1)
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis("off")
            
            axes[2].imshow(pred, cmap="coolwarm", vmin=0, vmax=1)
            axes[2].set_title("UNet-Mamba Prediction")
            axes[2].axis("off")
            
            axes[3].imshow(error_map)
            axes[3].set_title("Error Map (Green:TP, Red:FP, Yellow:FN)")
            axes[3].axis("off")
            
            plt.tight_layout()
            out_path = figures_dir / f"test_vis_{event_name}_{stem}_p{patch_idx}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            
    logger.info(f"Visualizations saved successfully to {figures_dir}")

if __name__ == "__main__":
    generate_visualizations()
