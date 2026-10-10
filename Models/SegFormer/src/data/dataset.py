"""PyTorch Dataset Implementation for FloodPlanet Semantic Segmentation.

Supports:
- Train mode with random 512x512 crop and spatial augmentations (Flips, Rotate90).
- Validation and Test modes with deterministic 512x512 tiling and tile coordinate bookkeeping.
- Label remapping via label_processing.map_labels.
- RGB extraction and ImageNet normalization.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import albumentations as A
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset

from src.data.label_processing import map_labels, TARGET_IGNORE
from src.data.preprocessing import extract_rgb, normalize_image, generate_tiles


class FloodPlanetDataset(Dataset):
    """PyTorch Dataset for FloodPlanet Inundation Semantic Segmentation."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        dataset_root: str = "Datasets",
        split: str = "train",
        input_size: int = 512,
        rgb_bands: Tuple[int, int, int] = (2, 1, 0),
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        ignore_index: int = TARGET_IGNORE,
        tile_overlap: int = 0,
    ):
        """
        Args:
            manifest_df: DataFrame containing sample rows (filename, image_path, mask_path, event, etc.).
            dataset_root: Root directory of FloodPlanet dataset.
            split: 'train', 'val', or 'test'.
            input_size: Patch crop size (512).
            rgb_bands: Band indices for RGB (default: (2, 1, 0)).
            image_mean: Normalization mean.
            image_std: Normalization standard deviation.
            ignore_index: Label integer code for ignored pixels (255).
            tile_overlap: Overlap between adjacent tiles in val/test modes.
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.rgb_bands = rgb_bands
        self.image_mean = image_mean
        self.image_std = image_std
        self.ignore_index = ignore_index
        self.tile_overlap = tile_overlap

        self.samples = manifest_df.to_dict("records")
        
        # Build tile index for deterministic evaluation (val/test)
        # Each 1024x1024 image produces 4 non-overlapping 512x512 tiles
        if self.split in ["val", "test"]:
            self.tiles_index = []
            for sample_idx, sample in enumerate(self.samples):
                w = sample.get("width", 1024)
                h = sample.get("height", 1024)
                tiles = generate_tiles((h, w), tile_size=self.input_size, overlap=self.tile_overlap)
                for t_idx, (ymin, xmin, ymax, xmax) in enumerate(tiles):
                    self.tiles_index.append({
                        "sample_idx": sample_idx,
                        "tile_idx": t_idx,
                        "ymin": ymin,
                        "xmin": xmin,
                        "ymax": ymax,
                        "xmax": xmax,
                        "total_tiles": len(tiles),
                    })
        else:
            self.tiles_index = None

        # Build augmentations for training
        if self.split == "train":
            self.transform = A.Compose([
                A.RandomCrop(height=self.input_size, width=self.input_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        if self.split in ["val", "test"]:
            return len(self.tiles_index)
        return len(self.samples)

    def _load_raw(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        img_rel = sample["image_path"]
        mask_rel = sample["mask_path"]
        img_full = os.path.join(self.dataset_root, img_rel)
        mask_full = os.path.join(self.dataset_root, mask_rel)

        raw_img = tifffile.imread(img_full)
        raw_mask = tifffile.imread(mask_full)

        # Extract RGB
        rgb_img = extract_rgb(raw_img, rgb_bands=self.rgb_bands)
        
        # Map raw label to training targets [0: non-flood, 1: flood, 255: ignore]
        target_mask = map_labels(raw_mask, ignore_index=self.ignore_index)
        
        return rgb_img, target_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.split in ["val", "test"]:
            tile_meta = self.tiles_index[idx]
            sample = self.samples[tile_meta["sample_idx"]]
            rgb_img, target_mask = self._load_raw(sample)

            ymin, xmin, ymax, xmax = tile_meta["ymin"], tile_meta["xmin"], tile_meta["ymax"], tile_meta["xmax"]
            crop_img = rgb_img[ymin:ymax, xmin:xmax]
            crop_mask = target_mask[ymin:ymax, xmin:xmax]

            # In case image border requires padding to reach input_size
            pad_h = self.input_size - crop_img.shape[0]
            pad_w = self.input_size - crop_img.shape[1]
            if pad_h > 0 or pad_w > 0:
                crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
                crop_mask = np.pad(crop_mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=self.ignore_index)

            # Normalize image
            norm_img = normalize_image(crop_img, mean=self.image_mean, std=self.image_std)
            tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()  # (3, H, W)
            tensor_mask = torch.from_numpy(crop_mask).long()  # (H, W)

            return {
                "image": tensor_img,
                "mask": tensor_mask,
                "sample_idx": tile_meta["sample_idx"],
                "tile_idx": tile_meta["tile_idx"],
                "coords": torch.tensor([ymin, xmin, ymax, xmax], dtype=torch.long),
                "event": sample["event"],
                "filename": sample["filename"],
            }

        else:
            sample = self.samples[idx]
            rgb_img, target_mask = self._load_raw(sample)

            # Apply training transforms
            augmented = self.transform(image=rgb_img, mask=target_mask)
            crop_img = augmented["image"]
            crop_mask = augmented["mask"]

            norm_img = normalize_image(crop_img, mean=self.image_mean, std=self.image_std)
            tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()  # (3, H, W)
            tensor_mask = torch.from_numpy(crop_mask).long()  # (H, W)

            return {
                "image": tensor_img,
                "mask": tensor_mask,
                "event": sample["event"],
                "filename": sample["filename"],
            }

    def get_full_sample(self, sample_idx: int) -> Dict[str, Any]:
        """Retrieve full uncropped image, mask, and tile coordinates for evaluation stitching."""
        sample = self.samples[sample_idx]
        rgb_img, target_mask = self._load_raw(sample)
        h, w = target_mask.shape[:2]
        tiles = generate_tiles((h, w), tile_size=self.input_size, overlap=self.tile_overlap)
        
        return {
            "rgb_image": rgb_img,
            "target_mask": target_mask,
            "tiles": tiles,
            "event": sample["event"],
            "filename": sample["filename"],
            "shape": (h, w),
        }
