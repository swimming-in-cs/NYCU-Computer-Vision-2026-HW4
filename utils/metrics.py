import torch
import math

def compute_psnr(pred, target, max_val=1.0):
    pred   = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    mse = torch.mean((pred - target) ** 2).item()
    return 100.0 if mse == 0 else 10 * math.log10(max_val ** 2 / mse)
