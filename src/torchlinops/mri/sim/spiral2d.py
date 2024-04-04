"""A simple spiral2d simulated MRI dataset"""
from dataclasses import dataclass, field
from typing import Tuple, Mapping, Optional

from einops import rearrange
import torch
import torch.nn as nn
import sigpy as sp
from torchlinops.mri._linops import NUFFT, SENSE

from ._trj import spiral_2d
from ._data import MRIDataset


@dataclass
class Spiral2dSimulatorConfig:
    im_size: Tuple[int, int]
    num_coils: int
    noise_std: float
    nufft_backend: str = "sigpy"
    spiral_2d_kwargs: Mapping = field(
        default_factory=lambda: {
            "n_shots": 16,
            "alpha": 1.5,
            "f_sampling": 0.4,
            "g_max": 0.04,
            "s_max": 100.0,
        }
    )


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

        # Linop
        self.A = self.make_linop(self.trj, self.mps)

    @property
    def data(self) -> MRIDataset:
        if self._data is None:
            ksp = self.A(self.img)
            ksp = ksp + self.config.noise_std * torch.randn_like(ksp)
            self._data = MRIDataset(
                trj=self.trj.data, mps=self.mps.data, ksp=ksp, img=self.img.data
            )
        return self._data

    def make_linop(self, trj: torch.Tensor, mps: torch.Tensor):
        S = SENSE(mps)
        F = NUFFT(
            trj,
            self.config.im_size,
            in_batch_shape=S.oshape[:-2],
            out_batch_shape=("R",),
            backend=self.config.nufft_backend,
        )
        return F @ S
