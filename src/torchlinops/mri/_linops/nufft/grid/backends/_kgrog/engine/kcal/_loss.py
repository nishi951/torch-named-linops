import torch
import torch.nn as nn


class VanillaGROGL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        B: Batch size
        C: Coil dimension
        npts: number of kernel points
        """
        pred_ksp = pred["ksp"]
        target_ksp = target["target_ksp"]  # [B C]
        diff = pred_ksp - target_ksp
        return torch.mean(torch.abs(diff.real) + torch.abs(diff.imag))
