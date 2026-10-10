import os
import argparse
import numpy as np
import torch
from pathlib import Path

try:
    import rasterio
except ImportError:
    import PIL.Image as rasterio

from config import load_config, BASE_DIR
from utils import get_logger
from model import build_model

def run_inference(image_path, checkpoint_path=None, output_path=None, config=None):
    if config is None:
        config = load_config()
    logger = get_logger("Inference")
    
    ckpt_dir = BASE_DIR / config['training']['save_dir']
    if checkpoint_path is None:
        checkpoint_path = ckpt_dir / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = ckpt_dir / "last_checkpoint.pth"
            
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading UNet-Mamba checkpoint from: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.eval()
    
    # Read input image
    image_path = Path(image_path)
    with rasterio.open(image_path) as src:
        img_data = src.read()  # (4, H, W)
        profile = src.profile if hasattr(src, 'profile') else None
        
    c, h, w = img_data.shape
    assert c == 4, f"Expected 4 input channels, got {c}"
    
    norm_mean = np.array(config['data']['norm_mean'], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config['data']['norm_std'], dtype=np.float32).reshape(4, 1, 1)
    
    img_norm = (img_data.astype(np.float32) - norm_mean) / (norm_std + 1e-6)
    
    patch_size = 512
    stride = 256
    
    # Tiled inference for large scenes
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    y_coords = list(range(0, max(1, h - patch_size + 1), stride))
    if y_coords[-1] + patch_size < h:
        y_coords.append(h - patch_size)
        
    x_coords = list(range(0, max(1, w - patch_size + 1), stride))
    if x_coords[-1] + patch_size < w:
        x_coords.append(w - patch_size)
        
    with torch.no_grad():
        for top in y_coords:
            for left in x_coords:
                patch = img_norm[:, top:top+patch_size, left:left+patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                
                logits = model(patch_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1, :, :].squeeze(0).cpu().numpy()
                
                prob_map[top:top+patch_size, left:left+patch_size] += probs
                count_map[top:top+patch_size, left:left+patch_size] += 1.0
                
    final_probs = prob_map / np.maximum(1.0, count_map)
    binary_pred = (final_probs >= 0.5).astype(np.uint8)
    
    if output_path is None:
        preds_dir = BASE_DIR / config['evaluation']['predictions_dir']
        os.makedirs(preds_dir, exist_ok=True)
        output_path = preds_dir / f"pred_{image_path.stem}.tif"
    else:
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
    if profile is not None:
        profile.update(count=1, dtype=rasterio.uint8)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(binary_pred, 1)
    else:
        np.savez_compressed(output_path.with_suffix('.npz'), prediction=binary_pred, probabilities=final_probs)
        
    logger.info(f"Inference completed! Binary prediction saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to 4-channel input satellite raster")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to UNet-Mamba checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output prediction raster path")
    args = parser.parse_args()
    
    run_inference(args.image, args.checkpoint, args.output)
