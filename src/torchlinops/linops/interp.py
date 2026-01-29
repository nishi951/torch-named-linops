from typing import Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import torchlinops.functional as F
from torchlinops.utils import default_to, default_to_dict
from .nameddim import ELLIPSES, NS, Shape
from .namedlinop import NamedLinop

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
        if locs_batch_shape is not None:
            if len(locs_batch_shape) > len(locs.shape) - 1:
                raise ValueError(
                    f"locs_batch_shape has length longer than batch dim of locs. locs_batch_shape: {locs_batch_shape}, locs: {locs.shape}"
                )
        batch_shape = default_to(("...",), batch_shape)
        locs_batch_shape = default_to(("...",), locs_batch_shape)
        grid_shape = default_to(("...",), grid_shape)
        shape = NS(batch_shape) + NS(grid_shape, locs_batch_shape)
        super().__init__(shape)
        self._shape.batch_shape = batch_shape
        self._shape.locs_batch_shape = locs_batch_shape
        self._shape.grid_shape = grid_shape
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

    @staticmethod
    def fn(interp, x, /):
        return F.interpolate(x, interp.locs, **interp._interp_params)

    @staticmethod
    def adj_fn(interp, x, /):
        return F.interpolate_adjoint(
            x, interp.locs, interp.grid_size, **interp._interp_params
        )

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
        return locs[tuple(locs_slc)]

    def size(self, dim):
        if dim in self._shape.locs_batch_shape:
            dim_idx = self._shape.locs_batch_shape.index(dim)
            return self.locs.shape[dim_idx]
        elif dim in self._shape.grid_shape:
            dim_idx = self._shape.grid_shape.index(dim)
            return self.grid_size[dim_idx]
        return None
