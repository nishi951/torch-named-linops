from copy import copy
from typing import Optional

import torch.fft as fft

from torchlinops.utils import default_to, cfftn, cifftn

from ..nameddim import NamedShape as NS, Shape, get_nd_shape
from .identity import Identity
from .namedlinop import NamedLinop


class FFT(NamedLinop):
    """$n$-dimensional Fast Fourier Transform as a named linear operator.

    Uses orthogonal normalization for true adjoint behavior.

    Attributes
    ----------
    ndim : int
        Number of spatial dimensions to transform.
    centered : bool
        Whether to treat the array center as the origin (sigpy convention).
    """

    def __init__(
        self,
        ndim: int,
        batch_shape: Optional[Shape] = None,
        grid_shapes: Optional[tuple[Shape, Shape]] = None,
        centered: bool = False,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of dimensions to transform (1, 2, or 3).
        batch_shape : Shape, optional
            Named batch dimensions prepended to the grid dimensions.
            Defaults to an empty shape.
        grid_shapes : tuple[Shape, Shape], optional
            Pair of shapes ``(primal, dual)`` naming the input (image-space)
            and output (k-space) grid dimensions. Defaults to
            ``(Nx[, Ny[, Nz]])`` and ``(Kx[, Ky[, Kz]])``.
        centered : bool, default False
            If ``True``, treat the center of the array (``N // 2``) as the
            origin. Mimics sigpy convention.
        """
        self.ndim = ndim
        self.dim = tuple(range(-self.ndim, 0))
        self.grid_shapes = grid_shapes
        if grid_shapes is None:
            grid_shapes = get_nd_shape(self.ndim), get_nd_shape(self.ndim, kspace=True)
        elif len(grid_shapes) != 2:
            raise ValueError(
                f"grid_shapes should consist of two shape tuples but got {grid_shapes}"
            )
        if len(grid_shapes[0]) != len(grid_shapes[1]):
            raise ValueError(
                f"Input and output shapes of FFT must have same length but got len({grid_shapes[0]} != len({grid_shapes[1]})"
            )
        batch_shape = default_to(("...",), batch_shape)
        dim_shape = NS(*grid_shapes)
        shape = NS(batch_shape) + dim_shape
        super().__init__(shape)
        self._shape.batch_shape = batch_shape
        self._shape.input_grid_shape = grid_shapes[0]
        self._shape.output_grid_shape = grid_shapes[1]
        self.centered = centered

    @property
    def batch_shape(self):
        return self._shape.batch_shape

    @staticmethod
    def fn(linop, x):
        if linop.centered:
            return cfftn(x, linop.dim, "ortho")
        return fft.fftn(x, dim=linop.dim, norm="ortho")

    @staticmethod
    def adj_fn(linop, x):
        if linop.centered:
            return cifftn(x, linop.dim, "ortho")
        return fft.ifftn(x, dim=linop.dim, norm="ortho")

    @staticmethod
    def normal_fn(linop, x):
        return x

    @staticmethod
    def split(linop, tile):
        """Splitting does nothing."""
        # TODO: raise an error if the FFT is split along an input or output grid dim
        new = copy(linop)
        return new

    def normal(self, inner=None):
        """Return the normal operator $F^H F$.

        With orthonormal normalization, $F^H F = I$, so this returns an
        ``Identity`` when no inner operator is provided.

        Parameters
        ----------
        inner : NamedLinop, optional
            Inner operator for Toeplitz embedding.

        Returns
        -------
        NamedLinop
            ``Identity`` if *inner* is ``None``, otherwise the composed normal.
        """
        if inner is None:
            return Identity(self.ishape)
        return super().normal(inner)
