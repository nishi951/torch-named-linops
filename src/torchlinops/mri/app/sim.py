from typing import Tuple

from easydict import EasyDict
from einops import rearrange
import torch
import sigpy as sp
import numpy as np

from torchlinops.core import fake_dims
from torchlinops.mri.linops import NUFFT, SENSE

from mr_sim.trajectory.trj import spiral_2d, tgas_spi

__all__ = [
    "Simulate",
]


class Simulate:
    """Mega-class for simulating all kinds of stuff"""

    def __init__(
        self,
        img: np.ndarray,
        trj: np.ndarray,
        mps: np.ndarray,
        noise_sigma: float = 0.0,
        device_idx: int = -1,
    ):
        """
        img: [... *im_size)]
        trj: [... K D]
        mps: [C *im_size]
        """
        self.img = img
        self.trj = trj
        self.mps = mps
        self.noise_sigma = noise_sigma
        self.device = torch.device(f"cuda:{device_idx}" if device_idx >= 0 else "cpu")

        # Derived
        self.im_size = self.mps.shape[1:]
        self.img_batch = img.shape[: -len(self.im_size)]
        self.trj_batch = self.trj.shape[:-2]
        F = NUFFT(
            torch.from_numpy(self.trj),
            self.im_size,
            in_batch_shape=fake_dims("A", len(self.img_batch)) + ("C",),
            out_batch_shape=fake_dims("B", len(self.trj_batch)),
        )
        S = SENSE(torch.from_numpy(self.mps))
        self.A = F @ S
        self.A.to(self.device)

    @classmethod
    def from_params(
        cls,
        im_size: Tuple = (64, 64),
        num_coils: int = 8,
    ):
        """Alternative constructor"""
        img = sp.shepp_logan(im_size).astype(np.complex64)

        if len(im_size) == 2:
            trj = spiral_2d(im_size)
            trj = rearrange(trj, "K R D -> R K D")
        elif len(im_size) == 3:
            trj = tgas_spi(im_size, ntr=500)
            trj = rearrange(trj, "K R T D -> R T K D")
        else:
            raise ValueError(
                f"Unsupported image dimension: {len(im_size)} (size {im_size})"
            )

        # Coils
        mps = sp.mri.birdcage_maps((num_coils, *im_size)).astype(np.complex64)
        return cls(img, trj, mps)

    def run(self):
        ksp = self.A(torch.from_numpy(self.img).to(self.device, dtype=torch.complex64))
        ksp = ksp + self.noise_sigma * torch.randn_like(ksp)
        return EasyDict(
            {
                "ksp": ksp.detach().cpu().numpy(),
                "trj": self.trj,
                "mps": self.mps,
                "img": self.img,
            }
        )
