"""A simple spiral2d simulated MRI dataset"""
from dataclasses import dataclass, field
from typing import Tuple, Mapping, Optional

from einops import rearrange
import torch
import torch.nn as nn
import sigpy as sp
from ._trj import spiral_2d
from torchlinops.mri._linops import NUFFT, SENSE


@dataclass
class Spiral2dSimulatorConfig:
    im_size: Tuple[int, int]
    num_coils: int
    noise_std: float
    spiral_2d_kwargs: Mapping = field(
        default_factory=lambda: {
            "n_shots": 16,
            "alpha": 1.5,
            "f_sampling": 0.4,
            "g_max": 40.0,
            "s_max": 100.0,
        }
    )


@dataclass
class MRIDataset:
    trj: torch.Tensor
    mps: torch.Tensor
    img: torch.Tensor
    ksp: torch.Tensor


class Spiral2dSimulator(nn.Module):
    def __init__(
        self,
        config: Spiral2dSimulatorConfig,
        img: Optional[torch.Tensor] = None,
        trj: Optional[torch.Tensor] = None,
        mps: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config

        self._data = None

        if img is None:
            img = sp.shepp_logan(self.config.im_size)
            img = torch.from_numpy(img).to(torch.complex64)
        self.img = nn.Parameter(img, requires_grad=False)

        if trj is None:
            trj = spiral_2d(self.config.im_size, **self.config.spiral_2d_kwargs)
            trj = torch.from_numpy(trj).to(torch.float32)
            trj = rearrange(trj, "K R D -> R K D")
        self.trj = nn.Parameter(trj, requires_grad=False)

        if mps is None:
            mps = sp.mri.birdcage_maps((self.config.num_coils, *self.config.im_size))
            mps = torch.from_numpy(mps).to(torch.complex64)
        self.mps = nn.Parameter(mps, requires_grad=False)

    @property
    def data(self) -> MRIDataset:
        if self._data is None:
            linop = self.linop(self.trj, self.mps)
            ksp = linop(self.img)
            ksp = ksp + self.config.noise_std * torch.randn_like(ksp)
            self._data = MRIDataset(self.trj.data, self.mps.data, self.img.data, ksp)
        return self._data

    def linop(self, trj: torch.Tensor, mps: torch.Tensor):
        S = SENSE(mps, in_batch_shape=("R",))
        F = NUFFT(
            trj,
            self.config.im_size,
            in_batch_shape=S.out_batch_shape,
            out_batch_shape=S.out_batch_shape,
        )
        return F @ S
