from typing import Optional

from copy import deepcopy
import torch.fft as fft

from .namedlinop import NamedLinop
from .nameddim import get_nd_shape, NS, Shape
from .identity import Identity


class FFT(NamedLinop):
    def __init__(
        self,
        ndim: int,
        batch_shape: Optional[Shape] = None,
        grid_shapes: Optional[Shape] = None,
        norm: str = "ortho",
        centered: bool = False,
    ):
        """
        Currently only supports 2D and 3D FFTs
        centered=True mimicks sigpy behavior

        batch_shape: NamedShape
        grid_shapes:


        """
        self.ndim = ndim
        self.dim = tuple(range(-self.ndim, 0))
        self.grid_shapes = grid_shapes
        if grid_shapes is None:
            dim_shape = NS(get_nd_shape(self.dim), get_nd_shape(self.dim, kspace=True))
        else:
            if len(grid_shapes) != 2:
                raise ValueError(
                    f"grid_shapes should consist of two shape tuples but got {grid_shapes}"
                )
            dim_shape = NS(*grid_shapes)
        shape = NS(batch_shape) + dim_shape
        super().__init__(shape)
        self._shape.add("batch_shape", batch_shape)
        self._shape.add("input_grid_shape", grid_shapes[0])
        self._shape.add("output_grid_shape", grid_shapes[1])
        self.norm = norm
        self.centered = centered

    @property
    def batch_shape(self):
        return self._shape.lookup("batch_shape")

    def forward(self, x, /):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x):
        if linop.centered:
            x = fft.ifftshift(x, dim=linop.dim)
        x = fft.fftn(x, dim=linop.dim, norm=linop.norm)
        if linop.centered:
            x = fft.fftshift(x, dim=linop.dim)
        return x

    @staticmethod
    def adj_fn(linop, x):
        if linop.centered:
            x = fft.ifftshift(x, dim=linop.dim)
        x = fft.ifftn(x, dim=linop.dim, norm=linop.norm)
        if linop.centered:
            x = fft.fftshift(x, dim=linop.dim)
        return x

    @staticmethod
    def normal_fn(linop, x):
        return x

    def split_forward(self, ibatch, obatch):
        new = type(self)(
            self.ndim,
            self.batch_shape,
            self.grid_shapes,
            self.norm,
            self.centered,
        )
        new._shape = deepcopy(self._shape)
        return new

    def normal(self, inner=None):
        if inner is None:
            return Identity(self.ishape)
        return super().normal(inner)
