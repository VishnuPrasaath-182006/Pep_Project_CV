import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from config import load_config, BASE_DIR
from transforms import SynchronizedAugmentations

class FloodPlanetPatchDataset(Dataset):
    def __init__(self, patch_csv_path, norm_mean=None, norm_std=None, is_train=True, preload=True):
        self.patch_csv_path = Path(patch_csv_path)
        if not self.patch_csv_path.exists():
            raise FileNotFoundError(f"Patch manifest not found: {self.patch_csv_path}")
            
        self.df_patches = pd.read_csv(self.patch_csv_path)
        self.is_train = is_train
        self.transform = SynchronizedAugmentations(is_train=is_train)
        
        # PlanetScope 4-band channel normalization statistics
        if norm_mean is None:
            self.norm_mean = np.array([2334.81, 2378.43, 2185.12, 3608.97], dtype=np.float32).reshape(4, 1, 1)
        else:
            self.norm_mean = np.array(norm_mean, dtype=np.float32).reshape(4, 1, 1)
            
        if norm_std is None:
            self.norm_std = np.array([1245.52, 1184.21, 1148.65, 1420.33], dtype=np.float32).reshape(4, 1, 1)
        else:
            self.norm_std = np.array(norm_std, dtype=np.float32).reshape(4, 1, 1)

        self.cache_in_memory = True
        self.cache = {}
        if preload:
            for idx in range(len(self.df_patches)):
                patch_path = self.df_patches.iloc[idx]["patch_path"]
                data = np.load(patch_path)
                self.cache[idx] = (data["image"].astype(np.float32), data["mask"].astype(np.int64))

    def __len__(self):
        return len(self.df_patches)

    def __getitem__(self, idx):
        if idx in self.cache:
            image, mask = self.cache[idx]
            image = image.copy()
            mask = mask.copy()
        else:
            row = self.df_patches.iloc[idx]
            patch_path = row["patch_path"]
            
            data = np.load(patch_path)
            image = data["image"].astype(np.float32)  # (4, 512, 512)
            mask = data["mask"].astype(np.int64)      # (512, 512)
            if self.cache_in_memory:
                self.cache[idx] = (image, mask)
            image = image.copy()
            mask = mask.copy()
        
        # Apply synchronized geometric augmentations if training
        image, mask = self.transform(image, mask)
        
        # Apply channel-wise z-score normalization
        image = (image - self.norm_mean) / (self.norm_std + 1e-6)
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor

def get_dataloaders(config=None, batch_size=None, num_workers=None):
    if config is None:
        config = load_config()
        
    processed_dir = BASE_DIR / config['data']['processed_dir']
    bs = batch_size if batch_size is not None else config['training']['batch_size']
    nw = num_workers if num_workers is not None else config['training']['num_workers']
    
    norm_mean = config['data']['norm_mean']
    norm_std = config['data']['norm_std']
    
    train_dataset = FloodPlanetPatchDataset(
        processed_dir / "train_patches.csv",
        norm_mean=norm_mean,
        norm_std=norm_std,
        is_train=True
    )
    val_dataset = FloodPlanetPatchDataset(
        processed_dir / "val_patches.csv",
        norm_mean=norm_mean,
        norm_std=norm_std,
        is_train=False
    )
    test_dataset = FloodPlanetPatchDataset(
        processed_dir / "test_patches.csv",
        norm_mean=norm_mean,
        norm_std=norm_std,
        is_train=False
    )
    
    # For fast in-memory RAM caching, set num_workers=0 to retain cache in main process
    nw = 0
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
