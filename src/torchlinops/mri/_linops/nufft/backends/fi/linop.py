from math import prod, sqrt
from typing import Optional, Tuple, Mapping
import logging

import torch
import cufinufft, finufft

from torchlinops.mri._linops.nufft.base import NUFFTBase
from ._flatten import multi_flatten
from . import functional as F
from . import planned as P
from .convert_trj import sp2fi, fi2sp

DEFAULT_UPSAMPFAC = 2.0

logger = logging.getLogger(__name__)


class FiNUFFT(NUFFTBase):
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
            self.upsampfac = extras["oversamp"]
        else:
            self.upsampfac = DEFAULT_UPSAMPFAC
        self.trj = sp2fi(self.trj, self.im_size)
        self.planned = False
        self._plans = []

    def change_im_size(self, new_im_size):
        self.im_size = new_im_size
        return self

    def forward(self, x: torch.Tensor):
        return self.fn(self, x, self.trj)

    def fn_noshared(
        self, x, trj, out=None, plan: Optional[P.FiNUFFTCombinedPlan] = None
    ):
        """
        x: [N... *im_size]
        trj: [K... D]

        Returns
        -------
        [N... K...] torch.Tensor

        """
        if plan is not None:
            out_ = P.nufft(x, plan, out)
        else:
            out_ = F.nufft(x, trj, out=out, upsampfac=self.upsampfac)
        if out is None:
            out = out_
        return out

    def adj_fn_noshared(
        self, y, trj, im_size, out=None, plan: Optional[P.FiNUFFTCombinedPlan] = None
    ):
        """
        y: [N... K...]
        trj: [K... D]

        Returns
        -------
        [N... *im_size] torch.Tensor
        """
        N_shape = y.shape[: -self.nK]
        oshape = (*N_shape, *self.im_size)
        if plan is not None:
            out_ = P.nufft_adjoint(y, plan, out)
        else:
            out_ = F.nufft_adjoint(y, trj, oshape, out=out, upsampfac=self.upsampfac)
        if out is None:
            out = out_
        return out

    @staticmethod
    def fn(linop, x, /, trj):
        """
        x: [[S...] N...  Nx Ny [Nz]] # A... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N... K...]
        """
        if linop.nS == 0:
            plan = linop._plans[0] if linop.planned else None
            return linop.fn_noshared(x, trj, plan=plan)
        assert (
            x.shape[: linop.shared_dims] == trj.shape[: linop.shared_dims]
        ), f"First {linop.shared_dims} dims of x, trj  must match but got x: {x.shape}, trj: {trj.shape}"
        S_shape = x.shape[: linop.nS]
        N_shape = x.shape[linop.nS : -linop.nD]
        K_shape = trj.shape[:-1]
        output_shape = (*S_shape, *N_shape, *K_shape)
        y = torch.zeros(
            (prod(S_shape), *N_shape, *K_shape),
            dtype=x.dtype,
            device=x.device,
        )
        trj, _ = multi_flatten(trj, linop.nS)
        for i in range(y.shape[0]):
            plan = linop._plans[i] if linop.planned else None
            y[i] = linop.fn_noshared(x[i], trj[i], plan=plan)
        y = torch.reshape(y, output_shape)
        return y

    @staticmethod
    def adj_fn(linop, y, /, trj):
        """
        y: [[S...] N... K...] # N... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N...  Nx Ny [Nz]]
        """
        N_shape = y.shape[linop.nS : -linop.nK]
        batch_oshape = (*N_shape, *linop.im_size)
        if linop.nS == 0:
            plan = linop._plans[0] if linop.planned else None
            return linop.adj_fn_noshared(y, trj, batch_oshape, plan=plan)

        assert (
            y.shape[: linop.shared_dims] == trj.shape[: linop.shared_dims]
        ), f"First {linop.shared_dims} dims of y, trj  must match but got y: {y.shape}, trj: {trj.shape}"
        S_shape = y.shape[: linop.nS]
        output_shape = (*S_shape, *N_shape, *linop.im_size)
        x = torch.zeros(
            (prod(S_shape), *N_shape, *linop.im_size),
            dtype=y.dtype,
            device=y.device,
        )
        trj, _ = multi_flatten(trj, linop.S)
        for i in range(x.shape[0]):
            plan = linop._plans[i] if linop.planned else None
            x[i] = linop.adj_fn_noshared(y[i], trj[i], batch_oshape, plan)
        x = torch.reshape(x, output_shape)
        return x

    @staticmethod
    def normal_fn(linop, x, /, trj):
        return linop.adj_fn(linop, linop.fn(linop, x, trj), trj)

    def split_forward(self, ibatch, obatch):
        """Override to undo effects of sp2fi"""
        new = type(self)(
            trj=self.split_forward_fn(ibatch, obatch, fi2sp(self.trj, self.im_size)),
            im_size=self.im_size,
            shared_batch_shape=self.shared_batch_shape,
            in_batch_shape=self.in_batch_shape,
            out_batch_shape=self.out_batch_shape,
            extras=self.extras,
            toeplitz=self.toeplitz,
            toeplitz_oversamp=self.toeplitz_oversamp,
        )
        if self.upsampfac != DEFAULT_UPSAMPFAC:
            new.upsampfac = self.upsampfac
        if self.planned:
            if (self.trj == new.trj).all():
                # Avoid unnecessary duplication of plans
                new._plans = self._plans
                new.planned = True
                new.plan_device = self.plan_device
                new.plan_type = self.plan_type
        return new

    def plan(self, device: torch.device = None):
        """Make some FiNUFFT plan objects ahead of time

        TODO: pull some of these arguments out and make them configurable
        TODO: Share some functionality with the Functional interface

        plan_type : str
            Either 'cpu' or 'gpu' or 'gpu:i' where i is the
            device index
        N_shape : int
            The shape of the input batch dimension. This is not typically known
            from the trajectory alone, so must be externally specified.
                Should also account for batching. E.g. if you have image
                batch dimensions (B1, B2), but you're batching over B1 with size 1,
                then N_shape should be (1, B2)
        """
        if "plan" not in self.extras:
            return self
        device = device if device is not None else torch.device("cpu")
        plan_type = str(device)
        N_shape = self.extras["plan"]["N_shape"]
        logger.info(
            f"Precomputing plans on device {plan_type}, with image batch {N_shape}"
        )
        if plan_type == "cpu":
            plan_backend = finufft
            kwargs = {
                "upsampfac": self.upsampfac,
                "spread_kerevalmeth": 1 if self.upsampfac == 2.0 else 0,
                "maxbatchsize": 1,  # For memory reasons
            }
            self.plan_device = torch.device("cpu")

        elif plan_type.startswith("cuda"):
            # 'gpu' or 'gpu:{index}'
            if ":" in plan_type:
                device_idx = int(plan_type.split(":")[1])
            else:
                device_idx = 0
            plan_backend = cufinufft
            kwargs = {
                "upsampfac": self.upsampfac,
                "gpu_kerevalmeth": 1 if self.upsampfac == 2.0 else 0,
                "gpu_maxbatchsize": 1,  # for memory reasons
            }
            self.plan_device = torch.device(f"cuda:{device_idx}")
        else:
            raise ValueError(f"Unrecognized plan type: {plan_type}")

        def makeplans(trj):
            """Helper function to quickly create new plans"""
            # Nufft type. 1=adjoint, 2=forward
            plan = P.FiNUFFTCombinedPlan(
                plan_backend.Plan(
                    2,
                    self.im_size,
                    prod(N_shape),
                    isign=-1,
                    dtype="complex64",
                    **kwargs,
                ),
                plan_backend.Plan(
                    1, self.im_size, prod(N_shape), isign=1, dtype="complex64", **kwargs
                ),
                im_size=self.im_size,
                N_shape=N_shape,
                K_shape=tuple(trj.shape[:-1]),
                plan_type=plan_type,
            )
            coords, _ = multi_flatten(trj, self.nK)
            coords = F.coord2contig(coords)
            if plan_type == "cpu":
                coords = tuple(c.detach().cpu().numpy() for c in coords)
            plan._forward.setpts(*coords)
            plan._adjoint.setpts(*coords)
            return plan

        if self.nS > 0:
            trj, _ = multi_flatten(self.trj, self.nS)
            for i in range(trj.shape[0]):
                self._plans.append(makeplans(trj[i]))
        else:
            self._plans.append(makeplans(self.trj))
        self.planned = True
        self.plan_type = plan_type
