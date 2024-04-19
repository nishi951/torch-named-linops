from math import prod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchlinops import NamedLinop
from torchlinops.mri._linops.nufft.base import NUFFTBase
from .indexing import multi_grid


__all__ = [
    "GriddedNUFFT",
]


class GriddedNUFFT(NUFFTBase):
    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        shared_batch_shape: Optional[Tuple] = None,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        *args,
        **kwargs,
    ):
        """
        img (input) [S... N... Nx Ny [Nz]]
        trj: [S... K..., D] integer-valued, in sigpy style [-N/2, N/2]
        in_batch_shape : Tuple
            The shape of [N...] in img
        out_batch_shape : Tuple
            The shape of [K...] in trj.
        shared_batch_shape : Tuple
            The shape of [S...] in trj
        """
        # Combine shared and in batch shape (distinction is no longer necessary)
        shared_batch_shape = (
            shared_batch_shape if shared_batch_shape is not None else tuple()
        )
        in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        super().__init__(
            trj,
            im_size,
            shared_batch_shape=shared_batch_shape,
            in_batch_shape=in_batch_shape,
            out_batch_shape=out_batch_shape,
            *args,
            **kwargs,
        )
        self.fft_dim = tuple(range(-self.nD, 0))
        # Convert to integer-valued for indexing
        self.trj = nn.Parameter(self.trj.data.long(), requires_grad=False)

    def change_im_size(self, new_im_size):
        # Necessary for sigpy scaling
        for i in range(self.trj.shape[-1]):
            self.trj[..., i] = (
                self.trj[..., i] * new_im_size[i] / self.im_size[i]
            ).long()
        self.im_size = new_im_size
        return self

    def forward(self, x: torch.Tensor):
        return self.fn(self, x, self.trj)

    @staticmethod
    def fn(linop, x, /, trj):
        """
        x: [[S...] N... Nx Ny [Nz]] # A... may include coils
        trj: [[S...] K... D] integer-valued index tensor
        output: [[S...] N... K...]
        """
        # FFT
        x = torch.fft.ifftshift(x, dim=linop.fft_dim)
        Fx = torch.fft.fftn(x, dim=linop.fft_dim, norm="ortho")

        # Index
        batch_slc = [slice(None)] * linop.nN
        if linop.nS == 0:
            trj_split = tuple(trj[..., i] for i in range(trj.shape[-1]))
            omega_slc = (*batch_slc, *trj_split)
            return Fx[omega_slc]
        S_shape = Fx.shape[: linop.nS]
        N_shape = Fx.shape[linop.nS : -linop.nD]
        K_shape = trj.shape[linop.nS : -1]
        output_shape = (*S_shape, *N_shape, *K_shape)
        Fx = torch.flatten(Fx, start_dim=0, end_dim=linop.nS - 1)
        trj = torch.flatten(trj, start_dim=0, end_dim=linop.nS - 1)
        out = torch.zeros(
            (prod(S_shape), *N_shape, *K_shape),
            dtype=Fx.dtype,
            device=Fx.device,
        )
        for s in range(out.shape[0]):
            trj_split = tuple(trj[s, ..., i] for i in range(trj.shape[-1]))
            omega_slc = (*batch_slc, *trj_split)  # [N... slice_x, slice_y, slice_z]
            out[s] = Fx[s][omega_slc]
        out = torch.reshape(out, output_shape)
        return out

    @staticmethod
    def adj_fn(linop, y, /, trj):
        """
        y: [[S...] N... K...]
        trj: [[S...] K... D] integer-valued index tensor
        output: [[S...] N... Nx Ny [Nz]]
        """
        # Un-Index (i.e. grid)
        if linop.nS == 0:
            Fx = multi_grid(y, linop.trj, final_size=linop.im_size)
        else:
            S_shape = y.shape[: linop.nS]
            N_shape = y.shape[linop.nS : -linop.nK]
            output_shape = (*S_shape, *N_shape, *linop.im_size)
            Fx = torch.zeros(
                (prod(S_shape), *N_shape, *linop.im_size),
                dtype=y.dtype,
                device=y.device,
            )
            y = torch.flatten(y, start_dim=0, end_dim=linop.nS - 1)
            trj = torch.flatten(trj, start_dim=0, end_dim=linop.nS - 1)
            for s in range(Fx.shape[0]):
                Fx[s] = multi_grid(y[s], trj[s], final_size=linop.im_size)
            Fx = torch.reshape(Fx, output_shape)

        # IFFT
        x = torch.fft.ifftn(Fx, dim=linop.fft_dim, norm="ortho")
        x = torch.fft.fftshift(x, dim=linop.fft_dim)
        return x

    @staticmethod
    def normal_fn(linop, x, /, trj):
        return linop.adj_fn(linop.fn(x, trj), trj)
