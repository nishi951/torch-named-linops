from math import prod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchlinops._core._linops import NamedLinop
from torchlinops._core._shapes import get2dor3d
from . import functional as F
from .convert_trj import sp2fi


class FiNUFFT(NamedLinop):
    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        shared_batch_shape: Optional[Tuple] = None,
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
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = (
            out_batch_shape if out_batch_shape is not None else tuple()
        )
        self.shared_batch_shape = (
            shared_batch_shape if shared_batch_shape is not None else tuple()
        )
        self.shared_dims = len(self.shared_batch_shape)
        ishape = self.shared_batch_shape + self.in_batch_shape + get2dor3d(im_size)
        oshape = self.shared_batch_shape + self.in_batch_shape + self.out_batch_shape
        super().__init__(ishape, oshape)
        self.trj = nn.Parameter(trj, requires_grad=False)
        self.im_size = im_size

        # Precompute
        self.D = len(im_size)

    def forward(self, x: torch.Tensor):
        return self.fn(x, self.trj)

    def fn(self, x, /, trj):
        """
        x: [[S...] N...  Nx Ny [Nz]] # A... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N... K...]
        """
        if self.shared_dims == 0:
            return F.nufft(x, sp2fi(trj, self.im_size))
        assert (
            x.shape[: self.shared_dims] == trj.shape[: self.shared_dims]
        ), f"First {self.shared_dims} dims of x, trj  must match but got x: {x.shape}, trj: {trj.shape}"
        S = x.shape[: self.shared_dims]
        x = torch.flatten(x, start_dim=0, end_dim=self.shared_dims - 1)
        trj = torch.flatten(trj, start_dim=0, end_dim=self.shared_dims - 1)
        N = x.shape[self.shared_dims : -self.D]
        K = trj.shape[self.shared_dims : -1]
        output_shape = (*S, *N, *K)
        y = torch.zeros((prod(S), *N, *K), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            F.nufft(x[i], sp2fi(trj[i], self.im_size), out=y[i])
        y = torch.reshape(y, output_shape)
        return y

    def adj_fn(self, y, /, trj):
        """
        y: [[S...] N... K...] # N... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N...  Nx Ny [Nz]]
        """
        if self.shared_dims == 0:
            return F.nufft_adjoint(y, sp2fi(trj, self.im_size), self.im_size)
        assert (
            y.shape[: self.shared_dims] == trj.shape[: self.shared_dims]
        ), f"First {self.shared_dims} dims of y, trj  must match but got y: {y.shape}, trj: {trj.shape}"
        S = y.shape[: self.shared_dims]
        y = torch.flatten(y, start_dim=0, end_dim=self.shared_dims)
        trj = torch.flatten(trj, start_dim=0, end_dim=self.shared_dims)
        N = y.shape[self.shared_dims : -self.D]
        oshape = (*N, *self.im_size)
        output_shape = (*S, *N, *self.im_size)
        x = torch.zeros((prod(S), *N, *self.im_size), dtype=y.dtype, device=y.device)
        for i in x.shape[0]:
            F.nufft_adjoint(y, sp2fi(trj, self.im_size), oshape, out=x[i])
        x = torch.reshape(x, output_shape)
        return x

    def normal_fn(self, x, /, trj):
        return self.adj_fn(self.fn(x, trj), trj)

    def split_forward(self, ibatch, obatch):
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.trj),
            im_size=self.im_size,
            in_batch_shape=self.in_batch_shape,
            out_batch_shape=self.out_batch_shape,
            shared_batch_shape=self.shared_batch_shape,
        )

    def split_forward_fn(self, ibatch, obatch, /, trj):
        shared_batch = obatch[: self.shared_dims]
        kbatch = obatch[self.shared_dims + len(self.in_batch_shape) :]
        trj_slc = tuple(shared_batch + kbatch + [slice(None)])
        # trj_slc = obatch[:-1] + [slice(None)] + obatch[-1:]
        return trj[trj_slc]

    def size(self, dim: str):
        return self.size_fn(dim, self.trj)

    def size_fn(self, dim: str, trj):
        if dim in self.shared_batch_shape:
            idx = self.shared_batch_shape.index(dim)
        elif dim in self.out_batch_shape:
            idx = len(self.shared_batch_shape) + self.out_batch_shape.index(dim)
        else:
            return None
        return trj.shape[idx]
