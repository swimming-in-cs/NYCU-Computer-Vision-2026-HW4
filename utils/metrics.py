"""Utility functions: PSNR computation."""

import torch
import math


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute PSNR between predicted and target tensors.
    Both should be float tensors in [0, max_val].
    """
    pred   = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * math.log10(max_val ** 2 / mse)
