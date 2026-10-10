"""SegFormer-B2 Architecture Implementation for Flood Inundation Segmentation.

Wraps Hugging Face SegformerForSemanticSegmentation with:
- MiT-B2 hierarchical Transformer backbone (nvidia/mit-b2).
- Two-class segmentation head (0: Non-flood, 1: Flood).
- Output logit interpolation to match input mask spatial resolution.
- Configurable CrossEntropyLoss with ignore_index=255.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.utils.logging import get_logger

logger = get_logger("SegFormerModel")


class SegFormerFloodModel(nn.Module):
    """SegFormer-B2 Semantic Segmentation Model for Binary Flood Detection."""

    def __init__(
        self,
        encoder_checkpoint: str = "nvidia/mit-b2",
        num_classes: int = 2,
        ignore_index: int = 255,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder_checkpoint = encoder_checkpoint
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if id2label is None:
            id2label = {0: "non_flood", 1: "flood"}
        if label2id is None:
            label2id = {"non_flood": 0, "flood": 1}

        self.id2label = id2label
        self.label2id = label2id

        if pretrained:
            logger.info(f"Loading pretrained SegFormer backbone from {encoder_checkpoint}...")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                encoder_checkpoint,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                semantic_loss_ignore_index=ignore_index,
                ignore_mismatched_sizes=True,
            )
        else:
            logger.info("Initializing SegFormer model with fresh random weights...")
            config = SegformerConfig.from_pretrained(
                encoder_checkpoint,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                semantic_loss_ignore_index=ignore_index,
            )
            self.model = SegformerForSemanticSegmentation(config)

        # Ensure correct loss ignore_index and do_reduce_labels
        self.model.config.semantic_loss_ignore_index = ignore_index
        self.model.config.num_labels = num_classes

        # Primary baseline loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            pixel_values: Input image tensor [B, 3, H, W].
            labels: Optional ground-truth mask tensor [B, H, W] with class IDs {0, 1, ignore_index}.
            
        Returns:
            Dictionary containing:
                'logits': Class logits resized to input resolution [B, num_classes, H, W].
                'loss': Scalar CrossEntropy loss (if labels provided).
        """
        input_h, input_w = pixel_values.shape[2:]
        
        # Raw model forward pass
        outputs = self.model(pixel_values=pixel_values)
        raw_logits = outputs.logits  # Shape: [B, num_classes, H/4, W/4]

        # Interpolate logits back to input spatial resolution
        upsampled_logits = F.interpolate(
            raw_logits,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )

        res: Dict[str, torch.Tensor] = {"logits": upsampled_logits}

        if labels is not None:
            loss = self.loss_fn(upsampled_logits, labels)
            res["loss"] = loss

        return res

    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Inference helper returning discrete integer class predictions [B, H, W]."""
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            preds = torch.argmax(outputs["logits"], dim=1)
        return preds


def build_segformer_model(
    model_name: str = "segformer_b2",
    num_classes: int = 2,
    ignore_index: int = 255,
    pretrained_checkpoint: str = "nvidia/mit-b2",
    pretrained: bool = True,
) -> SegFormerFloodModel:
    """Factory helper to instantiate a SegFormerFloodModel."""
    return SegFormerFloodModel(
        encoder_checkpoint=pretrained_checkpoint,
        num_classes=num_classes,
        ignore_index=ignore_index,
        pretrained=pretrained,
    )

