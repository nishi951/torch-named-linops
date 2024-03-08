"""A simple tgas_spi simulated MRI dataset"""
from dataclasses import dataclass, field
from typing import Tuple, Mapping, Optional

from einops import rearrange
import torch
import torch.nn as nn
import sigpy as sp
from ._trj import tgas_spi
from torchlinops.mri._linops import NUFFT, SENSE


@dataclass
class TGASSPISimulatorConfig:
    im_size: Tuple[int, int]
    num_coils: int
    num_TRs: int
    num_groups: int
    groups_undersamp: float
    noise_std: float
    spiral_2d_kwargs: Mapping = field(
        default_factory=lambda: {
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


class TGASSPISimulator(nn.Module):
    def __init__(
        self,
        config: TGASSPISimulatorConfig,
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
            trj = tgas_spi(
                self.config.im_size,
                self.config.num_TRs,
                self.config.num_groups,
                self.config.groups_undersamp,
                **self.config.spiral_2d_kwargs,
            )
            trj = torch.from_numpy(trj).to(torch.float32)
            trj = rearrange(trj, "K R T D -> R T K D")
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
        S = SENSE(mps)
        F = NUFFT(
            trj,
            self.config.im_size,
            in_batch_shape=S.out_batch_shape,
            out_batch_shape=S.out_batch_shape,
        )
        return F @ S
