import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTFrequencyLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft   = torch.fft.fft2(pred,   norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

class CombinedLoss(nn.Module):
    def __init__(self, lambda_fft=0.05):
        super().__init__()
        self.l1  = nn.L1Loss()
        self.fft = FFTFrequencyLoss()
        self.lambda_fft = lambda_fft
    def forward(self, pred, target):
        l1_loss  = self.l1(pred, target)
        fft_loss = self.fft(pred, target)
        return l1_loss + self.lambda_fft * fft_loss, l1_loss, fft_loss
