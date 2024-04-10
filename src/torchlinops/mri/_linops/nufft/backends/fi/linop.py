from math import prod
from typing import Optional, Tuple, Mapping

import torch
import cufinufft, finufft

from torchlinops.mri._linops.nufft.base import NUFFTBase
from . import functional as F
from .convert_trj import sp2fi, fi2sp

DEFAULT_UPSAMPFAC = 2.0

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
        self.plan = False
        self._plans = []
        self._adj_plans = []
        if extras is not None and "plan_ahead" in extras:
            self.make_plans(extras["plan_ahead"], extras["img_batch_size"])

    def change_im_size(self, new_im_size):
        self.im_size = new_im_size
        return self

    def forward(self, x: torch.Tensor):
        return self.fn(self, x, self.trj)

    def _plan_helper(self, plan, x, out=None):
        """Dealing with lots of edge cases"""
        if self.plan_type == "cpu":
            x = x.detach().cpu().numpy()
        if out is not None:
            plan.execute(x, out)
            return
        else:
            y = plan.execute(x)
        if self.plan_type == "cpu":
            return torch.from_numpy(y)
        return y

    @staticmethod
    def fn(linop, x, /, trj):
        """
        x: [[S...] N...  Nx Ny [Nz]] # A... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N... K...]
        """
        N = x.shape[linop.shared_dims : -linop.D]
        K = trj.shape[linop.shared_dims : -1]
        if linop.shared_dims == 0:
            if len(linop._plans) > 0:
                oshape = (*N, *K)
                y = linop._plan_helper(linop._plans[0], x)
                return torch.reshape(y, oshape)
            return F.nufft(x, trj, upsampfac=linop.upsampfac)
        assert (
            x.shape[: linop.shared_dims] == trj.shape[: linop.shared_dims]
        ), f"First {linop.shared_dims} dims of x, trj  must match but got x: {x.shape}, trj: {trj.shape}"
        S = x.shape[: linop.shared_dims]
        output_shape = (*S, *N, *K)
        x = torch.flatten(x, start_dim=0, end_dim=linop.shared_dims - 1)
        trj = torch.flatten(trj, start_dim=0, end_dim=linop.shared_dims - 1)
        y = torch.zeros((prod(S), *N, *K), dtype=x.dtype, device=x.device)
        for i in range(y.shape[0]):
            if len(linop._plans) > 0:
                linop._plan_helper(linop._plans[i], x[i], y[i])
            else:
                F.nufft(x[i], trj[i], out=y[i], upsampfac=linop.upsampfac)
        y = torch.reshape(y, output_shape)
        return y

    @staticmethod
    def adj_fn(linop, y, /, trj):
        """
        y: [[S...] N... K...] # N... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N...  Nx Ny [Nz]]
        """
        nK = len(linop.out_batch_shape)
        N = y.shape[linop.shared_dims : -nK]
        oshape = (*N, *linop.im_size)
        if linop.shared_dims == 0:
            if len(linop._adj_plans) > 0:
                # y = [N..., K...]
                if len(N) > 0:
                    y = torch.flatten(y, start_dim=0, end_dim=-(nK + 1))
                # y = [(N...), K...]
                y = torch.flatten(y, start_dim=-nK, end_dim=-1)
                # y = [(N...), (K...)]
                x = linop._plan_helper(linop._adj_plans[0], y)
                return x
            return F.nufft_adjoint(y, trj, oshape, upsampfac=linop.upsampfac)
        assert (
            y.shape[: linop.shared_dims] == trj.shape[: linop.shared_dims]
        ), f"First {linop.shared_dims} dims of y, trj  must match but got y: {y.shape}, trj: {trj.shape}"
        S = y.shape[: linop.shared_dims]
        N = y.shape[linop.shared_dims : -linop.D]
        oshape = (*N, *linop.im_size)
        output_shape = (*S, *N, *linop.im_size)
        y = torch.flatten(y, start_dim=0, end_dim=linop.shared_dims - 1)
        trj = torch.flatten(trj, start_dim=0, end_dim=linop.shared_dims - 1)
        x = torch.zeros((prod(S), *N, *linop.im_size), dtype=y.dtype, device=y.device)
        for i in range(x.shape[0]):
            if len(linop._adj_plans) > 0:
                linop._plan_helper(linop._adj_plans[i], y[i], x[i])
            else:
                F.nufft_adjoint(
                    y[i], trj[i], oshape, out=x[i], upsampfac=linop.upsampfac
                )
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
        if self.plan:
            if (self.trj == new.trj).all():
                # Zero-duplication of plans
                new._plans = self._plans
                new._adj_plans = self._adj_plans
                new.plan = self.plan
                new.plan_device = self.plan_device
                new.plan_type = self.plan_type
        return new

    def make_plans(self, plan_type: str, n_trans: int):
        """Make some FiNUFFT plan objects ahead of time

        TODO: pull some of these arguments out and make them configurable
        TODO: Share some functionality with the Functional interface

        plan_type : str
            Either 'cpu' or 'gpu' or 'gpu:i' where i is the
            device index
        n_trans : int
            The number of transforms to perform. This is not typically known
            from the trajectory alone, so must be externally specified.
                Should also account for batching. E.g. if you have image
                batch dimensions (B1, B2), but you're batching over B1 with size 1,
                then the number of transforms n_trans should be the size of B2.
        """
        if plan_type == "cpu":
            plan_backend = finufft
            kwargs = {
                "upsampfac": self.upsampfac,
                "spread_kerevalmeth": 1 if self.upsampfac == 2.0 else 0,
                "maxbatchsize": 1, # For memory reasons
            }
            self.plan_device = torch.device("cpu")

        elif plan_type.startswith("gpu"):
            # 'gpu' or 'gpu:{index}'
            if ":" in plan_type:
                device_idx = int(plan_type.split(":")[1])
            else:
                device_idx = 0
            plan_backend = cufinufft
            kwargs = {
                "upsampfac": self.upsampfac,
                "gpu_kerevalmeth": 1 if self.upsampfac == 2.0 else 0,
                "gpu_maxbatchsize": 1, # for memory reasons
            }
            self.plan_device = torch.device(f"cuda:{device_idx}")
        else:
            raise ValueError(f'Unrecognized plan type: {extras["plan_ahead"]}')

        # n_trans = prod(self.trj.shape[self.shared_dims : -self.D])
        def makeplans():
            """Helper function to quickly create new plans"""
            # Nufft type. 1=adjoint, 2=forward
            return (
                plan_backend.Plan(
                    2, self.im_size, n_trans, isign=-1, dtype="complex64", **kwargs
                ),
                plan_backend.Plan(
                    1, self.im_size, n_trans, isign=1, dtype="complex64", **kwargs
                ),
            )

        if self.shared_dims > 0:
            n_shared = prod(self.trj.shape[: self.shared_dims])
            trj_flat = torch.flatten(
                self.trj, start_dim=0, end_dim=self.shared_dims - 1
            )
            for i in range(n_shared):
                _plan, _adj_plan = makeplans()
                coords = torch.flatten(trj_flat[i], start_dim=0, end_dim=-2)
                coords = F.coord2contig(coords)
                if plan_type == "cpu":
                    flat_coords = tuple(c.detach().numpy() for c in flat_coords)
                else:
                    coords = tuple(c.to(self.plan_device) for c in coords)
                _plan.setpts(*coords)
                _adj_plan.setpts(*coords)
                self._plans.append(_plan)
                self._adj_plans.append(_adj_plan)
        else:
            trj_flat = torch.flatten
            _plan, _adj_plan = makeplans()
            coords = torch.flatten(self.trj, start_dim=0, end_dim=-2)
            coords = F.coord2contig(coords)
            if plan_type == "cpu":
                coords = tuple(c.detach().numpy() for c in coords)
            else:
                coords = tuple(c.to(self.plan_device) for c in coords)
            _plan.setpts(*coords)
            _adj_plan.setpts(*coords)
            self._plans.append(_plan)
            self._adj_plans.append(_adj_plan)
        self.plan = True
        self.plan_type = plan_type
