from functools import partial
from typing import Callable, Tuple, Optional
import logging

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ["PowerMethod", "angle_criterion", "diff_criterion"]


class PowerMethod(nn.Module):
    """Compute the maximum eigenvector and eigenvalue of a linear operator."""

    def __init__(
        self,
        num_iter: int,
        eps: float = 1e-6,
        stopping_criterion: Optional[Callable] = None,
    ):
        super(PowerMethod, self).__init__()
        self.num_iter = num_iter
        self.eps = eps
        self.stopping_criterion = (
            stopping_criterion
            if stopping_criterion is not None
            else partial(angle_criterion, threshold=1e-3)
        )

    def forward(self, A: Callable, ishape: Tuple, device: torch.device):
        logger.info("Computing maximum eigenvalue")

        # initialize random eigenvector directly on device
        v = torch.rand(ishape, dtype=torch.complex64, device=device)
        v, _ = normalize(v, self.eps)

        pbar = tqdm(range(self.num_iter), total=self.num_iter, desc="Max Eigenvalue")
        for _ in pbar:
            v_old = v.clone()
            v = A(v)
            v, vnorm = normalize(v)
            pbar.set_postfix({"eval": vnorm.item()})

            if self.stopping_criterion(v, v_old):
                break

        return v, vnorm


def inner(x: torch.Tensor, y: torch.Tensor):
    return torch.sum(torch.conj(x) * y)


def normalize(v: torch.Tensor, eps: float = 1e-6):
    vnorm = torch.linalg.norm(v)
    return v / torch.clamp(vnorm, min=eps), vnorm


def angle_criterion(v, v_old, threshold):
    angle = torch.arccos(torch.abs(inner(v_old, v)))
    logger.debug(f"Incremental angle change: {angle:0.3f}")
    return angle < threshold


def diff_criterion(v, v_old, threshold):
    diff = torch.sum(torch.abs(v - v_old) ** 2)
    logger.debug(f"|v - v_old|^2: {diff:0.3f}")
    return diff < threshold
