from typing import Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import torchlinops.functional as F
from torchlinops.utils import default_to, default_to_dict

from ..nameddim import ELLIPSES, NamedShape as NS, Shape
from .namedlinop import NamedLinop

__all__ = ["Interpolate"]


class Interpolate(NamedLinop):
    """Interpolate from a grid to a set of off-grid points.

    ```
    Input: (batch_shape, grid_shape)
    Output: (batch_shape, locs_batch_shape)
    ```

    Attributes
    ----------
    locs : nn.Parameter
        The target interpolation locations.
    grid_size : tuple[int, ...]
        The expected input grid size.
    interp_params : dict
        Dictionary of arguments for interpolation kernel.
    """

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
        norm: int = 1,
        pad_mode: str = "circular",
        kernel_params: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        locs : Float[Tensor, "... D"]
            The target interpolation locations, as a tensor of size (*locs_batch_size, num_dimensions).
            Uses 'ij' indexing.
        grid_size : tuple[int, ...]
            The expected input grid size. Should have same length as number of dimensions.
        batch_shape : Shape, optional
            The input/output batch shape. Defaults to "...".
        locs_batch_shape : Shape, optional
            The shape of the locs. Defaults to "...".
        grid_shape : Shape, optional
            The shape of the grid. Defaults to "...".
        width : float
            The width of the interpolation kernel.
        kernel : str
            The type of kernel to use. Current options are "kaiser_bessel" and "spline".
        norm : int
            The type of norm to use to measure distances. Current options are 1 and 2
        pad_mode : str
            The type of padding to apply.
        """
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
        self.interp_params = {
            "width": width,
            "kernel": kernel,
            "norm": norm,
            "pad_mode": pad_mode,
            "kernel_params": kernel_params,
        }

    @staticmethod
    def fn(interp, x, /):
        return F.interpolate(x, interp.locs, **interp.interp_params)

    @staticmethod
    def adj_fn(interp, x, /):
        return F.interpolate_adjoint(
            x, interp.locs, interp.grid_size, **interp.interp_params
        )

    def split_forward(self, ibatch, obatch):
        return type(self)(
            self.split_locs(ibatch, obatch, self.locs),
            self.grid_size,
            self._shape.batch_shape,
            self._shape.locs_batch_shape,
            self._shape.grid_shape,
            **self.interp_params,
        )

    def split_locs(self, ibatch, obatch, /, locs):
        """Can only split on locs dimensions"""
        if self._shape.locs_batch_shape == ELLIPSES:
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
