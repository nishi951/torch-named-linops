from dataclasses import dataclass, field
from typing import Tuple, Optional, Mapping

from einops import rearrange, repeat
import torch
import torch.nn as nn
import sigpy as sp


from torchlinops._core._linops import Dense, SumReduce
from torchlinops.mri._linops import SENSE, NUFFT
from ._data import SubspaceDataset
from .._trj import tgas_spi
from .qmri import brainweb

from .steady_state_simulator import (
    GREConfig,
    InversionRecoveryConfig,
    SteadyStateMRF
)

@dataclass
class TGASSPIMRFSimulatorConfig:
    im_size: Tuple[int, int, int]
    num_coils: int
    num_TRs: int
    num_groups: int
    num_bases: int
    groups_undersamp: float
    noise_std: float
    nufft_backend: str = 'fi'
    spiral_2d_kwargs: Mapping = field(
        default_factory=lambda: {
            "alpha": 1.5,
            "f_sampling": 0.4,
            "g_max": 40.0,
            "s_max": 100.0,
        }
    )


class TGASSPISubspaceMRFSimulator(nn.Module):
    def __init__(
        self,
        config: TGASSPIMRFSimulatorConfig,
        img: Optional[torch.Tensor] = None,
        trj: Optional[torch.Tensor] = None,
        mps: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        dic: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config

        self._data = None

        if img is None:
            img = brainweb.brainweb_phantom()
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

        if dic is None:
            ...
            # Load T1/T2 dictionary
            #

        self.dic = dic

        if phi is None:
            ...
        self.phi = nn.Parameter(phi, requires_grad=False)


        # Linop
        self.A = self.make_linop(self.trj, self.mps)

    @property
    def data(self) -> SubspaceDataset:
        if self._data is None:
            # Fully simulate data
            ksp = self.A(self.img)
            ksp = ksp + self.config.noise_std * torch.randn_like(ksp)
            self._data = SubspaceDataset(
                self.trj.data, self.mps.data, ksp, self.phi, self.dic, self.img.data
            )
        return self._data

    def make_linop(self, trj: torch.Tensor, mps: torch.Tensor, phi: torch.Tensor):
        S = SENSE(mps, in_batch_shape=("A",))
        F = NUFFT(
            trj,
            self.config.im_size,
            in_batch_shape=("A", "C"),
            out_batch_shape=("R", "K"),
            shared_batch_shape=("T",),
            backend=self.config.nufft_backend,
        )
        P = Diagonal(
            repeat(phi, "A T -> T A () () ()"), # Expand to match
            ioshape=("T", "A", "C", "R", "K"),
        )
        R = SumReduce(
            ishape=("T", "A", 'C', 'R', 'K'),
            oshape=("C", "R", 'T', 'K'),

        )
        return R @ P @ F @ S

    @staticmethod
    def get_default_simulator():
        FA, TR = brainweb.MRF_FISP()
        FA = torch.from_numpy(FA).to(torch.float32)
        TR = torch.from_numpy(TR).to(torch.float32)
        gre_config = GREConfig(
            flip_angle=FA,
            flip_angle_requires_grad=False,
            TR=TR,
            TR_requires_grad=False,
            TE=torch.tensor(0.7), # [ms]
            TE_requires_grad=False,
        )
        inv_rec_config = InversionRecoveryConfig(
            inversion_time=torch.tensor(20.),
            inversion_time_requires_grad=False,
            inversion_angle=torch.tensor(180.),
            inversion_angle_requires_grad=False,
            spoiler=False,
        )
        ssmrf_config = SteadyStateMRFSimulatorConfig(
            fisp_config=gre_config,
            inv_rec_config=inv_rec_config,
            wait_time=1500., # [ms]
            num_states=100,
            real_signal=True,
        )
        simulator = SteadyStateMRFSimulator(ssmrf_config)
        return simulator
