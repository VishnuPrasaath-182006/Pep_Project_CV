import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, smooth=0.05, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W) with values in {0, 1, ..., C-1} or ignore_index
        """
        num_classes = logits.shape[1]
        valid_mask = (target != self.ignore_index)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        target_clamped = target.clone()
        target_clamped[~valid_mask] = 0
        
        # One-hot encoding with label smoothing
        one_hot = F.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
        smooth_target = one_hot * (1.0 - self.smooth) + (self.smooth / num_classes)
        
        log_probs = F.log_softmax(logits.float(), dim=1)
        ce_loss = -(smooth_target * log_probs).sum(dim=1)  # (B, H, W)
        
        # Mask out invalid pixels
        loss = ce_loss[valid_mask].mean()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=0.05, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W)
        """
        num_classes = logits.shape[1]
        valid_mask = (target != self.ignore_index).unsqueeze(1) # (B, 1, H, W)
        
        if not valid_mask.any():
            return torch.zeros((), device=logits.device, requires_grad=True)
            
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0
        
        one_hot = F.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits.float(), dim=1)
        
        # Apply valid mask
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask
        
        # Compute Dice for each class across spatial dimensions
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        return dice_loss

class UnetMambaLoss(nn.Module):
    def __init__(self, soft_ce_smooth=0.05, dice_smooth=0.05, auxiliary_weight=0.4, ignore_index=-1):
        super().__init__()
        self.auxiliary_weight = auxiliary_weight
        self.soft_ce = SoftCrossEntropyLoss(smooth=soft_ce_smooth, ignore_index=ignore_index)
        self.dice = DiceLoss(smooth=dice_smooth, ignore_index=ignore_index)

    def _calc_loss(self, logits, target):
        loss_ce = self.soft_ce(logits, target)
        loss_dice = self.dice(logits, target)
        return loss_ce + loss_dice

    def forward(self, outputs, target):
        """
        outputs: tuple of (main_logits, aux_logits) during training, or main_logits during evaluation.
        target: (B, H, W)
        """
        if isinstance(outputs, (tuple, list)):
            main_logits, aux_logits = outputs[0], outputs[1]
            main_loss = self._calc_loss(main_logits, target)
            aux_loss = self._calc_loss(aux_logits, target)
            total_loss = main_loss + self.auxiliary_weight * aux_loss
            return total_loss
        else:
            return self._calc_loss(outputs, target)

def build_loss(config=None):
    if config is None:
        from config import load_config
        config = load_config()
    l_cfg = config['loss']
    return UnetMambaLoss(
        soft_ce_smooth=l_cfg.get('soft_ce_smooth', 0.05),
        dice_smooth=l_cfg.get('dice_smooth', 0.05),
        auxiliary_weight=l_cfg.get('auxiliary_weight', 0.4),
        ignore_index=l_cfg.get('ignore_index', -1)
    )
