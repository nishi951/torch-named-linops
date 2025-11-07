from copy import deepcopy
from typing import Optional

import torch.fft as fft

from .identity import Identity
from .nameddim import NS, Shape, get_nd_shape
from .namedlinop import NamedLinop


class FFT(NamedLinop):
    def __init__(
        self,
        ndim: int,
        batch_shape: Optional[Shape] = None,
        grid_shapes: Optional[tuple[Shape, Shape]] = None,
        norm: Optional[str] = "ortho",
        centered: bool = False,
    ):
        """
        Parameters
        ----------
        ndim : int
            Dimension of this fourier transform
        batch_shape: Shape, optional
            Batch dimensions, otherwise defaults to
        grid_shapes: (Shape, Shape), optional
            Pair of shapes corresponding to input (primal domain) and
            output (fourier/dual domain)
            Otherwise, defaults to (Nx[, Ny [, Nz]]) and (Kx[, Ky[, Kz]])
        norm : 'ortho' or None, default 'ortho'
            Normalization to apply to FFT.
            Note: technically only "ortho" results in a "true" forward/adjoint relationship.
        centered : bool, default False
            If True, treats the center of the array (i.e. N // 2) as the origin of k-space and image space
            If False, treats the array index origin (i.e. (0, 0) for 2D) as the origin of k-space and image space
            centered = True mimicks sigpy behavior.




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
        self._shape.batch_shape = batch_shape
        self._shape.input_grid_shape = grid_shapes[0]
        self._shape.output_grid_shape = grid_shapes[1]
        self.norm = norm
        self.centered = centered

    @property
    def batch_shape(self):
        return self._shape.batch_shape

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
