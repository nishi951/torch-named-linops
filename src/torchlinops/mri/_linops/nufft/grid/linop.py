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
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        shared_batch_shape: Optional[Tuple] = None,
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
        in_batch_shape = shared_batch_shape + in_batch_shape
        super().__init__(
            trj,
            im_size,
            shared_batch_shape=shared_batch_shape,
            in_batch_shape=in_batch_shape,
            out_batch_shape=out_batch_shape,
            *args,
            **kwargs,
        )
        self.fft_dim = tuple(range(-self.D, 0))
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
        return self.fn(x, self.trj)

    def fn(self, x, /, trj):
        """
        x: [N... Nx Ny [Nz]] # A... may include coils
        trj: [K... D] integer-valued index tensor
        output: [N... K...]
        """
        # FFT
        x = torch.fft.ifftshift(x, dim=self.fft_dim)
        Fx = torch.fft.fftn(x, dim=self.fft_dim, norm="ortho")

        # Index
        N = x.shape[: -self.D]
        batch_slc = (slice(None),) * len(N)
        trj_split = tuple(trj[..., i] for i in range(trj.shape[-1]))
        omega_slc = (*batch_slc, *trj_split)
        return Fx[omega_slc]

    def adj_fn(self, y, /, trj):
        """
        y: [N... K...]
        trj: [K... D] integer-valued index tensor
        output: [N... Nx Ny [Nz]]
        """
        # Un-Index (i.e. grid)
        Fx = multi_grid(y, self.trj, final_size=self.im_size)

        # IFFT
        x = torch.fft.ifftn(Fx, dim=self.fft_dim, norm="ortho")
        x = torch.fft.fftshift(x, dim=self.fft_dim)
        return x

    def normal_fn(self, x, /, trj):
        return self.adj_fn(self.fn(x, trj), trj)
