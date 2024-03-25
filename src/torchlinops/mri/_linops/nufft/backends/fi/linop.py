from math import prod
from typing import Optional, Tuple

import torch

from torchlinops.mri._linops.nufft.base import NUFFTBase
from . import functional as F
from .convert_trj import sp2fi


class FiNUFFT(NUFFTBase):
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
        trj: [S... K..., D] in sigpy style [-N/2, N/2]
        in_batch_shape : Tuple
            The shape of [N...] in img
        out_batch_shape : Tuple
            The shape of [K...] in trj.
        shared_batch_shape : Tuple
            The shape of [S...] in trj

        """
        super().__init__(
            trj,
            im_size,
            in_batch_shape,
            out_batch_shape,
            shared_batch_shape,
            *args,
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        return self.fn(x, self.trj)

    def fn(self, x, /, trj):
        """
        x: [[S...] N...  Nx Ny [Nz]] # A... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N... K...]
        """
        if self.shared_dims == 0:
            return F.nufft(x, sp2fi(trj.clone(), self.im_size))
        assert (
            x.shape[: self.shared_dims] == trj.shape[: self.shared_dims]
        ), f"First {self.shared_dims} dims of x, trj  must match but got x: {x.shape}, trj: {trj.shape}"
        S = x.shape[: self.shared_dims]
        N = x.shape[self.shared_dims : -self.D]
        K = trj.shape[self.shared_dims : -1]
        output_shape = (*S, *N, *K)
        x = torch.flatten(x, start_dim=0, end_dim=self.shared_dims - 1)
        trj = torch.flatten(trj, start_dim=0, end_dim=self.shared_dims - 1)
        y = torch.zeros((prod(S), *N, *K), dtype=x.dtype, device=x.device)
        for i in range(y.shape[0]):
            F.nufft(x[i], sp2fi(trj[i].clone(), self.im_size), out=y[i])
        y = torch.reshape(y, output_shape)
        return y

    def adj_fn(self, y, /, trj):
        """
        y: [[S...] N... K...] # N... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N...  Nx Ny [Nz]]
        """
        N = y.shape[self.shared_dims : -self.D]
        oshape = (*N, *self.im_size)
        if self.shared_dims == 0:
            return F.nufft_adjoint(y, sp2fi(trj.clone(), self.im_size), oshape)
        assert (
            y.shape[: self.shared_dims] == trj.shape[: self.shared_dims]
        ), f"First {self.shared_dims} dims of y, trj  must match but got y: {y.shape}, trj: {trj.shape}"
        S = y.shape[: self.shared_dims]
        N = y.shape[self.shared_dims : -self.D]
        oshape = (*N, *self.im_size)
        output_shape = (*S, *N, *self.im_size)
        y = torch.flatten(y, start_dim=0, end_dim=self.shared_dims - 1)
        trj = torch.flatten(trj, start_dim=0, end_dim=self.shared_dims - 1)
        x = torch.zeros((prod(S), *N, *self.im_size), dtype=y.dtype, device=y.device)
        for i in range(x.shape[0]):
            F.nufft_adjoint(y[i], sp2fi(trj[i].clone(), self.im_size), oshape, out=x[i])
        x = torch.reshape(x, output_shape)
        return x

    def normal_fn(self, x, /, trj):
        return self.adj_fn(self.fn(x, trj), trj)
