"""Label Processing Module for FloodPlanet Dataset.

Converts raw ground-truth raster annotations into standard semantic segmentation training targets:
- Raw 1 -> Target 0 (Non-flood / Background / Land)
- Raw 2 -> Target 1 (Flood / Inundated Water)
- Raw 0 / Nodata -> Target 255 (Ignore / Invalid)

Maintains strict separation between valid negative labels (non-flood) and ignored pixels.
"""

from typing import Union
import numpy as np
import torch


RAW_NODATA = 0
RAW_NON_FLOOD = 1
RAW_FLOOD = 2

TARGET_NON_FLOOD = 0
TARGET_FLOOD = 1
TARGET_IGNORE = 255


def map_labels(
    raw_mask: Union[np.ndarray, torch.Tensor],
    ignore_index: int = TARGET_IGNORE,
) -> Union[np.ndarray, torch.Tensor]:
    """Map raw FloodPlanet label values to training targets.
    
    Args:
        raw_mask: Array or Tensor of raw annotations (uint8, int32, or float32).
        ignore_index: Target integer code for nodata/unlabeled pixels (default: 255).
        
    Returns:
        Array or Tensor with classes:
            0: Non-flood
            1: Flood
            ignore_index: Ignore / invalid
            
    Raises:
        ValueError: If unexpected pixel values (outside {0, 1, 2}) are encountered.
    """
    if isinstance(raw_mask, np.ndarray):
        # Round any float annotations to nearest integer
        raw_int = np.rint(raw_mask).astype(np.int64)
        
        # Check for unexpected values
        valid_raw = {RAW_NODATA, RAW_NON_FLOOD, RAW_FLOOD}
        unique_vals = set(np.unique(raw_int).tolist())
        unexpected = unique_vals - valid_raw
        if unexpected:
            raise ValueError(f"Encountered unexpected raw label values: {unexpected}. Expected subset of {valid_raw}.")
            
        target_mask = np.full(raw_int.shape, fill_value=ignore_index, dtype=np.int64)
        target_mask[raw_int == RAW_NON_FLOOD] = TARGET_NON_FLOOD
        target_mask[raw_int == RAW_FLOOD] = TARGET_FLOOD
        target_mask[raw_int == RAW_NODATA] = ignore_index
        return target_mask
        
    elif isinstance(raw_mask, torch.Tensor):
        raw_int = torch.round(raw_mask).long()
        
        # Check unexpected values
        unique_vals = set(torch.unique(raw_int).cpu().numpy().tolist())
        valid_raw = {RAW_NODATA, RAW_NON_FLOOD, RAW_FLOOD}
        unexpected = unique_vals - valid_raw
        if unexpected:
            raise ValueError(f"Encountered unexpected raw label values: {unexpected}. Expected subset of {valid_raw}.")
            
        target_mask = torch.full_like(raw_int, fill_value=ignore_index, dtype=torch.long)
        target_mask[raw_int == RAW_NON_FLOOD] = TARGET_NON_FLOOD
        target_mask[raw_int == RAW_FLOOD] = TARGET_FLOOD
        target_mask[raw_int == RAW_NODATA] = ignore_index
        return target_mask
        
    else:
        raise TypeError(f"Unsupported mask type: {type(raw_mask)}. Expected numpy.ndarray or torch.Tensor.")


def validate_label_tensor(
    label_tensor: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = TARGET_IGNORE,
) -> bool:
    """Validate that a label tensor contains only valid class IDs or ignore_index."""
    valid_classes = set(range(num_classes)) | {ignore_index}
    unique_classes = set(torch.unique(label_tensor).cpu().numpy().tolist())
    invalid = unique_classes - valid_classes
    if invalid:
        raise ValueError(f"Target mask contains invalid class IDs: {invalid}. Allowed: {valid_classes}")
    return True
