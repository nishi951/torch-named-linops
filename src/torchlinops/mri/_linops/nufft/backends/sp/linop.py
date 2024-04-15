from math import prod
from typing import Optional, Tuple, Mapping

import torch

from torchlinops.mri._linops.nufft.base import NUFFTBase
from . import functional as F


class SigpyNUFFT(NUFFTBase):
    """NUFFT with Sigpy backend"""

    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        shared_batch_shape: Optional[Tuple] = None,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        extras: Optional[Mapping] = None,
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
            shared_batch_shape=shared_batch_shape,
            in_batch_shape=in_batch_shape,
            out_batch_shape=out_batch_shape,
            extras=extras,
            *args,
            **kwargs,
        )
        if extras is not None and "oversamp" in extras:
            self.oversamp = extras["oversamp"]
        else:
            self.oversamp = 1.25
        if extras is not None and "width" in extras:
            self.width = extras["width"]
        else:
            self.width = 4

    def forward(self, x: torch.Tensor):
        return self.fn(self, x, self.trj)

    @staticmethod
    def fn(linop, x, /, trj):
        """
        x: [[S...] N...  Nx Ny [Nz]] # A... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N... K...]
        """
        if linop.nS == 0:
            return F.nufft(x, trj, linop.oversamp, linop.width)
        assert (
            x.shape[: linop.nS] == trj.shape[: linop.nS]
        ), f"First {linop.shared_dims} dims of x, trj  must match but got x: {x.shape}, trj: {trj.shape}"

        S_shape = x.shape[: linop.nS]
        N_shape = x.shape[linop.nS : -linop.nD]
        K_shape = trj.shape[linop.nS : -1]
        x = torch.flatten(x, start_dim=0, end_dim=linop.nS)
        trj = torch.flatten(trj, start_dim=0, end_dim=linop.nS)

        # x, _ = multi_flatten(x, (linop.nS, linop.nN))
        # trj, _ = multi_flatten(trj, (linop.nS, linop.nK))
        output_shape = (*S_shape, *N_shape, *K_shape)
        y = torch.zeros(
            (prod(S_shape), *N_shape, *K_shape), dtype=x.dtype, device=x.device
        )
        for i in range(x.shape[0]):
            y[i] = F.nufft(x[i], trj[i], linop.oversamp, linop.width)
        y = torch.reshape(y, output_shape)
        return y

    @staticmethod
    def adj_fn(linop, y, /, trj):
        """

        y: [[S...] N... K...]
        trj: [K... D], in [-im_size//2, im_size//2]
        output: [N... Nx Ny [Nz]]
        """
        if linop.nS == 0:
            N = y.shape[: -linop.nK]
            oshape = (*N, *linop.im_size)
            x = F.nufft_adjoint(y, trj, oshape, linop.oversamp, linop.width)
            return x
        assert (
            y.shape[: linop.nS] == trj.shape[: linop.nS]
        ), f"First {linop.nS} dims of y, trj  must match but got y: {y.shape}, trj: {trj.shape}"
        S_shape = y.shape[: linop.nS]
        N_shape = y.shape[linop.nS : -linop.D]
        oshape = (*N, *linop.im_size)
        output_shape = (*S_shape, *N_shape, *linop.im_size)
        y = torch.flatten(y, start_dim=0, end_dim=linop.nS)
        trj = torch.flatten(trj, start_dim=0, end_dim=linop.nS)
        # y, _ = multi_flatten(y, (linop.nS, linop.nN, linop.nK))
        # trj, _ = multi_flatten(trj, (linop.nS, linop.nK))

        x = torch.zeros(
            (prod(S_shape), *N_shape, *linop.im_size),
            dtype=y.dtype,
            device=y.device,
        )
        for i in range(x.shape[0]):
            x[i] = F.nufft_adjoint(y[i], trj[i], oshape, linop.oversamp, linop.width)
        x = torch.reshape(x, output_shape)
        return x

    @staticmethod
    def normal_fn(self, x, /, trj):
        return self.adj_fn(self.fn(x, trj), trj)
