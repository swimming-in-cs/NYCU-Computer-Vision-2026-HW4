"""
Loss functions for image restoration.

Includes:
- L1 Loss (pixel-level)
- FFT Frequency Loss (Modification #2: frequency-domain supervision)
- Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class FFTFrequencyLoss(nn.Module):
    """
    Modification #2: Frequency-domain loss.
    Computes L1 loss on the FFT magnitude spectrum.
    Encourages the model to recover high-frequency details (edges, textures).

    Reference: Focal Frequency Loss (ICCV 2021)
    """
    def forward(self, pred, target):
        # FFT over spatial dims
        pred_fft   = torch.fft.fft2(pred,   norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        pred_mag   = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return F.l1_loss(pred_mag, target_mag)


class CombinedLoss(nn.Module):
    """
    L1 + lambda_fft * FFT loss.
    Default lambda_fft = 0.05 (small weight to avoid over-sharpening).
    """
    def __init__(self, lambda_fft=0.05):
        super().__init__()
        self.l1  = L1Loss()
        self.fft = FFTFrequencyLoss()
        self.lambda_fft = lambda_fft

    def forward(self, pred, target):
        l1_loss  = self.l1(pred, target)
        fft_loss = self.fft(pred, target)
        return l1_loss + self.lambda_fft * fft_loss, l1_loss, fft_loss
