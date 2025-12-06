"""
utils.py

Shared utilities: seeding, Dice metric, PyTorch Dataset wrapper.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Compute Dice coefficient for binary masks.
    preds, targets: tensors of shape (N, 1, H, W), logits or probabilities.
    """
    # If logits, convert to probabilities
    if preds.dtype.is_floating_point:
        preds = torch.sigmoid(preds)

    preds_bin = (preds > 0.5).float()
    targets = targets.float()

    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    dice = (2.0 * intersection + eps) / union
    return dice.mean().item()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    """
    Combination of BCEWithLogitsLoss and soft Dice loss.
    """
    targets = targets.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6
    dice = (2.0 * intersection + 1e-6) / union
    dice_loss = 1.0 - dice.mean()

    return bce_weight * bce + (1.0 - bce_weight) * dice_loss


class WildfireTileDataset(Dataset):
    """
    Simple PyTorch Dataset for segmentation:
        X: (N, C, H, W)
        Y: (N, 1, H, W)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.ndim == 4, "X should be (N, C, H, W)"
        assert Y.ndim == 4, "Y should be (N, 1, H, W)"
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y
