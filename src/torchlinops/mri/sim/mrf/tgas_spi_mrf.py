from dataclasses import dataclass, field
from math import prod
from typing import Tuple, Optional, Mapping

from einops import rearrange, repeat, einsum
import torch
import torch.nn as nn
import sigpy as sp
import numpy as np
from tqdm import tqdm


from torchlinops._core._tiling import Batch
from torchlinops._core._linops import SumReduce, Diagonal
from torchlinops.utils import batch_tqdm, ordinal
from torchlinops.mri._linops import SENSE, NUFFT, DCF
from ._data import SubspaceDataset
from .._trj import tgas_spi
from .qmri import brainweb

from .steady_state_simulator import (
    GREConfig,
    InversionRecoveryConfig,
    SteadyStateMRFSimulator,
    SteadyStateMRFSimulatorConfig,
)

__all__ = ["TGASSPISubspaceMRFSimulatorConfig", "TGASSPISubspaceMRFSimulator"]


@dataclass
class TGASSPISubspaceMRFSimulatorConfig:
    im_size: Tuple[int, int, int]
    num_coils: int
    num_TRs: int
    num_groups: int
    num_bases: int
    groups_undersamp: float
    noise_std: float
    voxel_batch_size: int
    tr_batch_size: int
    coil_batch_size: int
    nufft_backend: str = "fi"
    nufft_extras: Optional[Mapping] = None
    spiral_2d_kwargs: Mapping = field(
        default_factory=lambda: {
            "alpha": 1.5,
            "f_sampling": 0.4,
            "g_max": 40.0,
            "s_max": 100.0,
        }
    )
    debug: bool = False


class TGASSPISubspaceMRFSimulator(nn.Module):
    def __init__(
        self,
        config: TGASSPISubspaceMRFSimulatorConfig,
        device: Optional[torch.device] = None,
        q_img: Optional[Mapping[str, torch.Tensor]] = None,
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
        q_img : Mapping[str, torch.Tensor]
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
        self.simulator = self.make_simulator()

        # Quantitative phantom
        if q_img is None:
            data = brainweb.brainweb_phantom()
            img_t1 = torch.from_numpy(data["t1"]).float()
            img_t1 = nn.Parameter(img_t1, requires_grad=False)
            img_t2 = torch.from_numpy(data["t2"]).float()
            img_t2 = nn.Parameter(img_t2, requires_grad=False)
            img_pd = torch.from_numpy(data["pd"]).float()
            img_pd = nn.Parameter(img_pd, requires_grad=False)
            q_img = {"img_t1": img_t1, "img_t2": img_t2, "img_pd": img_pd}
            if self.config.debug:
                w = 32
                slc = tuple(
                    slice(
                        self.config.im_size[i] // 2 - w // 2,
                        self.config.im_size[i] // 2 + w // 2,
                    )
                    for i in range(len(self.config.im_size))
                )
                q_img["img_t1"] = q_img["img_t1"][slc]
                q_img["img_t2"] = q_img["img_t2"][slc]
                q_img["img_pd"] = q_img["img_pd"][slc]
                self.config.im_size = q_img["img_t1"].shape
        self.q_img = nn.ParameterDict(q_img)
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
            trj = rearrange(trj, "K R T D -> T R K D")
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
            pd = torch.tensor([1.0])  # needs to be 1D
        self.pd = pd
        self.t1t2pd = nn.ParameterList(
            [
                nn.Parameter(p, requires_grad=False)
                for p in torch.meshgrid((self.t1, self.t2, self.pd), indexing="ij")
            ]
        )

        # Use GPU if available for next steps
        self.device = device
        self.to(self.device)

        self.dic = self.simulator(*(t.flatten() for t in self.t1t2pd))
        self.dic = self.dic.reshape(*self.t1t2pd[0].shape, -1)  # [T1, T2, PD, TR]
        self.dic = nn.Parameter(self.dic, requires_grad=False)
        # Temporal subspace
        if phi is None:
            phi = self.compress_dictionary(self.dic, n_coeffs=self.config.num_bases)
        self.phi = nn.Parameter(phi, requires_grad=False)

        # Linops
        self.Asim = self.make_simulation_linop(self.trj, self.mps)
        self.A = self.make_subspace_linop(self.trj, self.mps, self.phi)

    def convert_simulator_to_graph(self, simulator, device, batch_size):
        """Uses CUDA graphs to speed up (?) the simulation
        Currently broken

        """
        static_t1 = torch.zeros(batch_size, device=device, requires_grad=True)
        static_t2 = torch.zeros(batch_size, device=device, requires_grad=True)
        static_pd = torch.zeros(batch_size, device=device, requires_grad=True)
        # output = simulator(static_t1, static_t2, static_pd)
        sim_graph = torch.cuda.make_graphed_callables(
            simulator, (static_t1, static_t2, static_pd)
        )
        return sim_graph

    def simulate(self) -> SubspaceDataset:
        if self._data is None:
            device = self.q_img["img_t1"].device
            # TODO: Get graphs to work?Fully simulate data
            # if not device == torch.device('cpu'):
            #     simulator = self.convert_simulator_to_graph(self.simulator, device, self.config.voxel_batch_size)
            # else:
            #     simulator = self.simulator
            simulator = self.simulator
            # Too big to store in memory:
            ksp_size = tuple(self.Asim.size(dim) for dim in self.Asim.oshape)
            ksp = torch.zeros(*ksp_size, dtype=torch.complex64, device=device)

            with torch.no_grad():  # Very important to avoid memory blowups
                # Compute per-voxel signal
                spatiotemporal_image = torch.zeros(
                    (prod(self.config.im_size), self.config.num_TRs), device="cpu"
                )
                img_t1 = self.q_img["img_t1"].flatten()
                img_t2 = self.q_img["img_t2"].flatten()
                img_pd = self.q_img["img_pd"].flatten()
                for vstart, vend in batch_tqdm(
                    total=prod(self.config.im_size),
                    batch_size=self.config.voxel_batch_size,
                    desc="Spatiotemporal Voxel Simulation",
                ):
                    spatiotemporal_image[vstart:vend] = simulator(
                        img_t1[vstart:vend],
                        img_t2[vstart:vend],
                        img_pd[vstart:vend],
                    )
                spatiotemporal_image = spatiotemporal_image.reshape(
                    *self.config.im_size, self.config.num_TRs
                )
                spatiotemporal_image = rearrange(
                    spatiotemporal_image, "... T -> T ..."
                ).contiguous()

                # Compute ksp signal
                batched_Asim = Batch(
                    self.Asim,
                    input_device=device,
                    output_device=device,
                    output_dtype=torch.complex64,
                    pbar=True,
                    T=self.config.tr_batch_size,
                    C=self.config.coil_batch_size,
                )
                ksp = batched_Asim(spatiotemporal_image)
                # for tstart, tend in tqdm(batch_iterator(total=self.config.num_TRs,
                #                                         batch_size=self.config.tr_batch_size),
                #                          total=self.config.num_TRs // self.config.tr_batch_size,
                #                          desc='Temporal batching'):

                #     spatiotemporal_batch = spatiotemporal_image[tstart:tend].to(device)
                #     ksp[tstart:tend] = self.Asim(spatiotemporal_batch)

            self.t_img = spatiotemporal_image
            # Project to subspace
            self.sub_img = einsum(
                self.phi.conj(), self.t_img.to(device), "A T, T ... -> A ..."
            )
            ksp = ksp + self.config.noise_std * torch.randn_like(ksp)
            self._data = SubspaceDataset(
                self.trj.data,
                self.mps.data,
                ksp,
                self.phi,
                self.q_img,
                self.t_img,
                self.sub_img,
                self.dic,
                self.t1,
                self.t2,
            )
        return self._data

    def make_simulation_linop(self, trj, mps):
        S = SENSE(mps, in_batch_shape=("T",))
        F = NUFFT(
            trj,
            self.config.im_size,
            in_batch_shape=("C",),
            out_batch_shape=("R", "K"),
            shared_batch_shape=("T",),
            backend=self.config.nufft_backend,
            extras=self.config.nufft_extras,
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
            out_batch_shape=("T", "R", "K"),
            toeplitz=True,
            backend=self.config.nufft_backend,
        )
        P = Diagonal(
            repeat(phi, "A T -> A () T () ()"),  # Expand to match
            ioshape=("A", "C", "T", "R", "K"),
        )
        R = SumReduce(
            ishape=("A", "C", "T", "R", "K"),
            oshape=("C", "T", "R", "K"),
        )
        D = DCF(
            trj,
            self.config.im_size,
            ioshape=("C", "T", "R", "K"),
            device_idx=ordinal(self.device),
        )
        return (D ** (1 / 2)) @ R @ P @ F @ S

    @staticmethod
    def make_simulator():
        FA, TR = brainweb.MRF_FISP()
        FA = torch.from_numpy(FA).to(torch.float32)
        TR = torch.from_numpy(TR).to(torch.float32)
        TE = torch.empty_like(TR).fill_(0.7)  # [ms]
        gre_config = GREConfig(
            flip_angle=FA,
            flip_angle_requires_grad=False,
            TR=TR,
            TR_requires_grad=False,
            TE=TE,
            TE_requires_grad=False,
        )
        inv_rec_config = InversionRecoveryConfig(
            TI=torch.tensor(20.0),
            TI_requires_grad=False,
            inv_angle=torch.tensor(180.0),
            inv_angle_requires_grad=False,
        )
        ssmrf_config = SteadyStateMRFSimulatorConfig(
            fisp_config=gre_config,
            inv_rec_config=inv_rec_config,
            wait_time=1500.0,  # [ms]
            num_states=10,
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
