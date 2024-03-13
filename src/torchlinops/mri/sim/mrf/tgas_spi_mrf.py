from dataclasses import dataclass, field
from typing import Tuple, Optional, Mapping

from einops import rearrange, repeat
import torch
import torch.nn as nn
import sigpy as sp
import numpy as np


from torchlinops._core._linops import SumReduce, Diagonal
from torchlinops.mri._linops import SENSE, NUFFT
from ._data import SubspaceDataset
from .._trj import tgas_spi
from .qmri import brainweb

from .steady_state_simulator import (
    GREConfig,
    InversionRecoveryConfig,
    SteadyStateMRFSimulator,
    SteadyStateMRFSimulatorConfig,
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
    nufft_backend: str = "fi"
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
        qimg: Optional[Mapping[str, torch.Tensor]] = None,
        trj: Optional[torch.Tensor] = None,
        mps: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        t1: Optional[torch.Tensor] = None,
        t2: Optional[torch.Tensor] = None,
        pd: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        config : TGASSPIRMRFSimulatorConfig
            Configuration object for this simulator

        All following args are optional:
        qimg : Mapping[str, torch.Tensor]
            Quantitative phantom image expressed as a mapping from 't1', 't2', 'pd' to their respective maps
        trj : torch.Tensor
            [T K... D] kspace trajectory, sigpy scaling [-N/2, N/2]
        mps : torch.Tensor
            Sensitivity maps
        phi : torch.Tensor
            [n_bases, T] Temporal subspace []
        t1, t2, pd : torch.Tensor
            T1, T2, and Proton Density arrays
        """
        super().__init__()
        self.config = config
        self._data = None
        self._simulator = None

        # Quantitative phantom
        if qimg is None:
            data = brainweb.brainweb_phantom()
            t1 = torch.from_numpy(data["t1"]).float()
            t1 = nn.Parameter(t1, requires_grad=False)
            t2 = torch.from_numpy(data["t2"]).float()
            t2 = nn.Parameter(t2, requires_grad=False)
            pd = torch.from_numpy(data["pd"]).float()
            pd = nn.Parameter(pd, requires_grad=False)
            qimg = {"t1": t1, "t2": t2, "pd": pd}
        self.qimg = nn.ParameterDict(qimg)

        # Trajectory
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

        # Sensitivity maps
        if mps is None:
            mps = sp.mri.birdcage_maps((self.config.num_coils, *self.config.im_size))
            mps = torch.from_numpy(mps).to(torch.complex64)
        self.mps = nn.Parameter(mps, requires_grad=False)

        # Tissue parameters and dictionary
        if t1 is None:
            t1 = self.tgas_t1()
        self.t1 = t1
        if t2 is None:
            t2 = self.tgas_t2()
        self.t2 = t2
        if pd is None:
            pd = torch.tensor(1.0)
        self.pd = pd
        self.t1t2pd = torch.meshgrid((self.T1, self.T2, self.pd), indexing="ij")

        # Temporal subspace
        if phi is None:
            phi = self.compress_dictionary(self.dic, n_coeffs=5)
        self.phi = nn.Parameter(phi, requires_grad=False)

        # Linops
        self.Asim = self.make_simulation_linop(self.trj, self.mps)
        self.A = self.make_subspace_linop(self.trj, self.mps, self.phi)

    @property
    def dic(self):
        """Put as a property to allow for a .to(device) call"""
        if self._dic is None:
            self._dic = self.simulator(*self.t1t2pd)
        return self._dic

    @property
    def data(self) -> SubspaceDataset:
        if self._data is None:
            # Fully simulate data
            spatiotemporal_image = self.simulator(
                self.qimg["t1"], self.qimg["t2"], self.qimg["pd"]
            )

            ksp = self.Asim(spatiotemporal_image)

            ksp = ksp + self.config.noise_std * torch.randn_like(ksp)
            self._data = SubspaceDataset(
                self.trj.data,
                self.mps.data,
                ksp,
                self.phi,
                self.qimg,
                dict(self.dic),
                self.t1,
                self.t2,
            )
        return self._data

    def make_simulation_linop(self, trj, mps):
        S = SENSE(mps, in_batch_shape=("T",))
        F = NUFFT(
            trj,
            self.config.im_size,
            in_batch_shape=("T", "C"),
            out_batch_shape=("R", "K"),
            shared_batch_shape=("T",),
            backend=self.config.nufft_backend,
        )
        return F @ S

    def make_subspace_linop(
        self, trj: torch.Tensor, mps: torch.Tensor, phi: torch.Tensor
    ):
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
            repeat(phi, "A T -> T A () () ()"),  # Expand to match
            ioshape=("T", "A", "C", "R", "K"),
        )
        R = SumReduce(
            ishape=("T", "A", "C", "R", "K"),
            oshape=("C", "R", "T", "K"),
        )
        return R @ P @ F @ S

    @staticmethod
    def make_simulator():
        FA, TR = brainweb.MRF_FISP()
        FA = torch.from_numpy(FA).to(torch.float32)
        TR = torch.from_numpy(TR).to(torch.float32)
        gre_config = GREConfig(
            flip_angle=FA,
            flip_angle_requires_grad=False,
            TR=TR,
            TR_requires_grad=False,
            TE=torch.tensor(0.7),  # [ms]
            TE_requires_grad=False,
        )
        inv_rec_config = InversionRecoveryConfig(
            inversion_time=torch.tensor(20.0),
            inversion_time_requires_grad=False,
            inversion_angle=torch.tensor(180.0),
            inversion_angle_requires_grad=False,
            spoiler=False,
        )
        ssmrf_config = SteadyStateMRFSimulatorConfig(
            fisp_config=gre_config,
            inv_rec_config=inv_rec_config,
            wait_time=1500.0,  # [ms]
            num_states=100,
            real_signal=True,
        )
        simulator = SteadyStateMRFSimulator(ssmrf_config)
        return simulator

    @staticmethod
    def tgas_t1():
        """
        T1, T2 values specified in ms
        """
        T1 = np.concatenate(
            (np.array(range(20, 3001, 20)), np.array(range(3200, 5001, 200)))
        )
        return torch.from_numpy(T1).float()

    @staticmethod
    def tgas_t2():
        T2 = np.concatenate(
            (
                np.array(range(10, 201, 2)),
                np.array(range(220, 1001, 20)),
                np.array(range(1050, 2001, 50)),
                np.array(range(2100, 4001, 100)),
            )
        )
        return torch.from_numpy(T2).float()

    @staticmethod
    def compress_dictionary(dic: torch.Tensor, n_coeffs: int):
        """Given a dictionary, find the low-rank temporal basis that captures it
        dic: [... T]

        Returns:
        phi: [n_coeffs T]
        """
        dic = rearrange(dic, "... t -> (...) t")
        # Normalize the dictionary along the time dimension
        dic /= torch.linalg.norm(dic, axis=-1, keepdims=True)
        _, _, Vh = torch.linalg.svd(dic, full_matrices=False)
        phi = Vh[:n_coeffs]
        return phi
