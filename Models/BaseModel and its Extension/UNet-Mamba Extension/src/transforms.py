import random
import torch
import numpy as np

class SynchronizedAugmentations:
    """
    Synchronized geometric augmentations for 4-channel satellite images and segmentation masks.
    Only authorized transformations: Horizontal Flip, Vertical Flip, and 90-degree Rotations.
    """
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, image, mask):
        """
        image: numpy array (C, H, W)
        mask: numpy array (H, W)
        """
        if not self.is_train:
            return image, mask
            
        # Horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
            
        # Vertical flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
            
        # Random 90, 180, 270 degree rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
            
        return image, mask
