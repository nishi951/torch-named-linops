"""
Implementations of various algorithms.
Written by Christopher M. Sandino (sandino@stanford.edu), 2020.
Modified by Mark Nishimura, 2023
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

import torch
from torch import nn
from tqdm import tqdm


__all__ = [
    "CGHparams",
    "ConjugateGradient",
]


@dataclass
class CGHparams:
    num_iter: int
    """Number of iterations"""


class ConjugateGradient(nn.Module):
    """
    Implementation of conjugate gradient algorithm to invert
    the linear system:
        y = A x
    where A is a symmetric, positive-definite matrix.
    In multi-coil MRI reconstruction, A is not symmetric. However, we can
    form the normal equations to get the problem in the form above:
        A^T y = A^T A x
    Based on code by Jon Tamir.
    """

    def __init__(self, A, hparams):
        super().__init__()
        self.A = A
        self.hparams = hparams

        self.rs = None  # For holding residuals
        self.xs = None  # For holding intermediate results

    def zdot(self, x1, x2):
        """
        Complex dot product between tensors x1 and x2.
        """
        return torch.sum(x1.conj() * x2)

    def zdot_single(self, x):
        """
        Complex dot product between tensor x and itself
        """
        return self.zdot(x, x).real

    def update(self, x, p, r, rsold):
        # Compute step size
        logger.debug("Computing step size alpha...")
        Ap = self.A(p)
        pAp = self.zdot(p, Ap)
        alpha = rsold / pAp

        # Take step
        logger.debug("Taking step...")
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = self.zdot_single(r)
        beta = rsnew / rsold
        rsold = rsnew
        p = beta * p + r

        return x, p, r, rsold

    def forward(self, y: torch.Tensor, x_init: Optional[torch.Tensor] = None):
        """
        Solves Ax = y, where A is positive semidefinite
        x: initial guess at solution
        y: RHS input data
        """
        # Initialize variables
        logger.debug("Computing initial residual...")
        if x_init is not None:
            x = x_init.clone()
            r = y - self.A(x)
        else:
            x = torch.zeros_like(y)
            r = y
        rsold = self.zdot_single(r)
        p = r
        self.rs = [rsold.item()]
        self.xs = [x]
        for i in tqdm(
            range(self.hparams.num_iter),
            desc="Conjugate Gradient",
            leave=False,
        ):
            logger.debug(f"CG Iteration {i}")
            x, p, r, rsold = self.update(x, p, r, rsold)
            logger.debug(f"New residual: {rsold}")
            self.rs.append(rsold.item())
            self.xs.append(x.clone().detach().cpu().numpy())
        return x
