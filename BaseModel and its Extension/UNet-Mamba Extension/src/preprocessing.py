import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import rasterio
except ImportError:
    import PIL.Image as rasterio

from config import load_config, BASE_DIR
from utils import get_logger

def preprocess_dataset(config=None, output_dir=None):
    if config is None:
        config = load_config()
    logger = get_logger("Preprocessing")
    
    patch_size = config['data']['patch_size']  # 512
    stride = config['data']['stride']          # 256
    
    manifests_dir = BASE_DIR / config['data']['manifests_dir']
    if output_dir is None:
        processed_dir = BASE_DIR / config['data']['processed_dir']
    else:
        processed_dir = Path(output_dir)
        
    os.makedirs(processed_dir, exist_ok=True)
    
    # Store channel normalization sums for training statistics
    sum_channels = np.zeros(4, dtype=np.float64)
    sqsum_channels = np.zeros(4, dtype=np.float64)
    total_pixels = 0
    
    patch_manifests = {}
    
    for split_name in ["train", "val", "test"]:
        split_manifest_path = manifests_dir / f"{split_name}.csv"
        if not split_manifest_path.exists():
            logger.error(f"Manifest not found: {split_manifest_path}. Run split_manager first!")
            continue
            
        df_split = pd.read_csv(split_manifest_path)
        split_out_dir = processed_dir / split_name
        os.makedirs(split_out_dir, exist_ok=True)
        
        patch_records = []
        logger.info(f"Preprocessing {split_name} split ({len(df_split)} scenes, patch_size={patch_size}, stride={stride})...")
        
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Processing {split_name}"):
            event_name = row["event"]
            stem = row["stem"]
            img_path = row["img_path"]
            mask_path = row["mask_path"]
            
            with rasterio.open(img_path) as s_img:
                img_data = s_img.read()  # (4, H, W)
            with rasterio.open(mask_path) as s_mask:
                mask_raw = s_mask.read(1)  # (H, W)
                
            # Convert mask values: 1 -> 0 (BG), 2 -> 1 (Flood), others -> -1 (Ignore)
            mask_mapped = np.full_like(mask_raw, fill_value=-1, dtype=np.int64)
            mask_mapped[mask_raw == 1] = 0
            mask_mapped[mask_raw == 2] = 1
            
            # Compute stats on train images
            if split_name == "train":
                valid_mask = (mask_mapped != -1)
                for c in range(4):
                    band_vals = img_data[c][valid_mask].astype(np.float64)
                    sum_channels[c] += np.sum(band_vals)
                    sqsum_channels[c] += np.sum(band_vals**2)
                total_pixels += np.sum(valid_mask)
                
            c, h, w = img_data.shape
            
            # Sliding window coordinates
            y_coords = list(range(0, h - patch_size + 1, stride))
            if (h - patch_size) % stride != 0 and y_coords[-1] != h - patch_size:
                y_coords.append(h - patch_size)
                
            x_coords = list(range(0, w - patch_size + 1, stride))
            if (w - patch_size) % stride != 0 and x_coords[-1] != w - patch_size:
                x_coords.append(w - patch_size)
                
            patch_idx = 0
            for top in y_coords:
                for left in x_coords:
                    img_patch = img_data[:, top:top+patch_size, left:left+patch_size]
                    mask_patch = mask_mapped[top:top+patch_size, left:left+patch_size]
                    
                    patch_filename = f"{event_name}_{stem}_p{patch_idx}_{top}_{left}.npz"
                    patch_filepath = split_out_dir / patch_filename
                    
                    # Save patch compressed numpy array
                    np.savez_compressed(
                        patch_filepath,
                        image=img_patch.astype(np.uint16),
                        mask=mask_patch.astype(np.int64)
                    )
                    
                    patch_records.append({
                        "event": event_name,
                        "stem": stem,
                        "patch_idx": patch_idx,
                        "top": top,
                        "left": left,
                        "height": patch_size,
                        "width": patch_size,
                        "patch_path": str(patch_filepath),
                        "split": split_name
                    })
                    patch_idx += 1
                    
        df_patches = pd.DataFrame(patch_records)
        patch_manifest_path = processed_dir / f"{split_name}_patches.csv"
        df_patches.to_csv(patch_manifest_path, index=False)
        patch_manifests[split_name] = len(df_patches)
        logger.info(f"Saved {split_name} patches manifest ({len(df_patches)} patches) to {patch_manifest_path}")
        
    if total_pixels > 0:
        mean_channels = (sum_channels / total_pixels).tolist()
        var_channels = (sqsum_channels / total_pixels) - (np.array(mean_channels)**2)
        std_channels = np.sqrt(np.maximum(1e-6, var_channels)).tolist()
        
        logger.info(f"Computed Train Normalization Stats:")
        logger.info(f"Mean (Bands 1..4): {[round(x, 2) for x in mean_channels]}")
        logger.info(f"Std (Bands 1..4):  {[round(x, 2) for x in std_channels]}")
        
    logger.info(f"Preprocessing completed! Total patches: {patch_manifests}")

if __name__ == "__main__":
    preprocess_dataset()
