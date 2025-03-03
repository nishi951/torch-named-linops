from typing import Optional
from jaxtyping import Float
from torch import Tensor

from warnings import warn

import torch.nn as nn

import torchlinops.functional as F
from torchlinops.utils import default_to, default_to_dict

from .namedlinop import NamedLinop
from .nameddim import ELLIPSES, NS, Shape

__all__ = ["Interpolate"]


class Interpolate(NamedLinop):
    def __init__(
        self,
        locs: Float[Tensor, "... D"],
        grid_size: tuple[int, ...],
        batch_shape: Optional[Shape] = None,
        locs_batch_shape: Optional[Shape] = None,
        grid_shape: Optional[Shape] = None,
        # Interp params
        width: float = 4.0,
        kernel: str = "kaiser_bessel",
        norm: str = "1",
        pad_mode: str = "circular",
        kernel_params: Optional[dict] = None,
    ):
        batch_shape = default_to(("...",), batch_shape)
        locs_batch_shape = default_to(("...",), locs_batch_shape)
        grid_shape = default_to(("...",), grid_shape)
        shape = NS(batch_shape) + NS(grid_shape, locs_batch_shape)
        super().__init__(shape)
        self._shape.add("batch_shape", batch_shape)
        self._shape.add("locs_batch_shape", locs_batch_shape)
        self._shape.add("grid_shape", grid_shape)
        self.locs = nn.Parameter(locs, requires_grad=False)
        self.grid_size = grid_size

        # Do this here instead of repeating it in both fn() and adjoint_fn()
        kernel_params = default_to_dict(dict(beta=1.0), kernel_params)
        self._interp_params = {
            "width": width,
            "kernel": kernel,
            "norm": norm,
            "pad_mode": pad_mode,
            "kernel_params": kernel_params,
        }

    def forward(self, x):
        return self.fn(self, x, self.locs)

    @staticmethod
    def fn(linop, x, /, locs):
        return F.interpolate(x, locs, **linop._interp_params)

    @staticmethod
    def adj_fn(linop, x, /, locs):
        return F.interpolate_adjoint(x, locs, linop.grid_size, **linop._interp_params)

    @staticmethod
    def normal_fn(linop, x, /, locs):
        return linop.adj_fn(linop, linop.fn(linop, x, locs), locs)
        # return linop.adj_fn(linop, linop.fn(linop, x, locs), locs)

    def split_forward(self, ibatch, obatch):
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.locs),
            self.grid_size,
            self._shape.batch_shape,
            self._shape.locs_batch_shape,
            self._shape.grid_shape,
            **self._interp_params,
        )

    def split_forward_fn(self, ibatch, obatch, /, locs):
        """Can only split on locs dimensions"""
        if self._shape.locs_batch_shape == ELLIPSES:
            # warn(f"Attempted to split {self.__class__.__name__} but ")
            return locs

        N = len(self._shape.locs_batch_shape)
        locs_slc = []
        for oslc in obatch[-N:]:
            locs_slc.append(oslc)
        locs_slc.append(slice(None))
        return locs[locs_slc]
