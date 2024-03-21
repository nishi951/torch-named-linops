from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.fft as fft

from .base import NUFFTBase
from . import functional as F

__all__ = [
    "GriddedNUFFT",
]


class GriddedNUFFT(NUFFTBase):
    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        shared_batch_shape: Optional[Tuple] = None,
        nufft_kwargs: Optional[Mapping] = None,
    ):
        """
        img (input) [S... N... Nx Ny [Nz]]
        trj: [S... K..., D] gridded to integers in sigpy style [-N//2, N//2]
        in_batch_shape : Tuple
            The shape of [N...] in img
        out_batch_shape : Tuple
            The shape of [K...] in trj.
        shared_batch_shape : Tuple
            The shape of [S...] in trj

        """

        super().__init__(trj, im_size, in_batch_shape, out_batch_shape, shared_batch_shape, nufft_kwargs)
        self.fft_dim = list(range(-self.D, 0))

    def forward(self, x: torch.Tensor):
        return self.fn(x, self.trj)

    def fn(self, x, /, trj):
        """
        x: [[S...] N...  Nx Ny [Nz]] # A... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N... K...]
        """
        if self.shared_dims == 0:
            return F.gridded_nufft(x, trj, self.fft_dim)

        # FFT
        x = fft.ifftshift(x, dim=self.fft_dim)
        Fx = fft.fftn(x, dim=self.fft_dim, norm="ortho")

        # Index
        A = x.shape[: -self.D]
        batch_slc = (slice(None),) * len(A)
        trj_split = tuple(trj[..., i] for i in range(trj.shape[-1]))
        omega_slc = (*batch_slc, *trj_split)
        return Fx[omega_slc]

    def adj_fn(self, y, /, trj):
        """
        y: [A... B... K]
        trj: [B... K D] integer-valued index tensor
        output: [A... Nx Ny [Nz]]
        """
        if self.shared_dims == 0:
            return F.gridded_nufft_adjoint(y, trj, self.fft_dim, self.im_size)
        return x

    def
