"""NUFFT Base class"""

from copy import copy

import torch
from jaxtyping import Float
from torch import Tensor, nn

from torchlinops.utils import default_to, default_to_dict

from ...nameddim import (
    Shape,
    get_nd_shape,
)
from ...nameddim import (
    NamedDimension as ND,
)
from ..chain import Chain
from ..namedlinop import NamedLinop
from ..pad_last import Pad
from .utils import scale_int


class NUFFTBase(Chain):
    """Non-uniform Fast Fourier Transform (type II) as a named linear operator."""

    def __init__(
        self,
        locs: Float[Tensor, "... D"],
        grid_size: tuple[int, ...],
        output_shape: Shape,
        input_shape: Shape | None = None,
        input_kshape: Shape | None = None,
        batch_shape: Shape | None = None,
        **options,
    ):
        """
        Parameters
        ----------
        locs : Tensor, float
            Shape [... D] Tensor where last dimension is the spatial dimension.
            locs[..., i] Should be in the range [-N//2, N//2] where N is the grid_size[i], i.e.
            the grid size associated with that dimension
        grid_size : tuple of ints
            The expected spatial dimension of the input tensor.
        output_shape : Shape
        input_shape : Shape, optional
        input_kshape : Shape, optional
        batch_shape : Shape, optional
            NUFFT is implemented as a chain of padding, FFT, and interpolation
            Named Dimensions are set as follows:

            Pad: (*batch_shape, *input_shape) -> (*batch_shape, *next_unused(input_shape))
            FFT: (*batch_shape, *next_unused(input_shape)) -> (*batch_shape, *input_kshape)
            Interp: (*batch_shape, *input_kshape) -> (*batch_shape, *output_shape)


        """
        # Useful parameters to save
        self.locs = locs
        self.grid_size = grid_size
        self.options = default_to_dict(self.default_options, options)
        self._init_shapes(
            grid_size, output_shape, input_shape, input_kshape, batch_shape
        )
        linops = self.build()
        super().__init__(*linops, name=type(self).__name__)
        self.post_init_hook()

    def build(self) -> list[NamedLinop]:
        """Main builder function"""
        raise NotImplementedError()

    def post_init_hook(self):
        """Post-setup actions for after __init__ is called."""

    # Init helper methods
    def _init_shapes(
        self,
        grid_size,
        output_shape,
        input_shape,
        input_kshape,
        batch_shape,
    ):
        self.grid_size = grid_size
        ndim = len(self.grid_size)
        self.input_shape = ND.infer(default_to(get_nd_shape(ndim), input_shape))
        self.input_kshape = ND.infer(
            default_to(get_nd_shape(ndim, kspace=True), input_kshape)
        )
        self.output_shape = ND.infer(output_shape)
        self.batch_shape = ND.infer(default_to(("...",), batch_shape))

    def adjoint(self):
        # Hybrid of chain adjoint and namedlinop adjoint
        adj = copy(self)
        adj._shape = adj._shape.H

        linops = list(linop.adjoint() for linop in reversed(self.linops))
        adj.linops = nn.ModuleList(linops)
        return adj

    def normal(self, inner=None):
        if self.options.get("toeplitz", False):
            from .toeplitz import toeplitz_psf  # Avoid circular import

            dtype = self.options.get("toeplitz_dtype")
            oversamp = self.options.get("toeplitz_oversamp", 2.0)
            toep_kernel = toeplitz_psf(self, inner, dtype=dtype, oversamp=oversamp)
            pad = Pad(
                scale_int(self.grid_size, oversamp),
                self.grid_size,
                in_shape=self.input_shape,
                batch_shape=self.batch_shape,
            )
            fft = self.fft
            return pad.normal(fft.normal(toep_kernel))
        return super().normal(inner)

    @staticmethod
    def beta(width, oversamp):
        """
        https://sigpy.readthedocs.io/en/latest/_modules/sigpy/fourier.html#nufft

        References
        ----------
        Beatty PJ, Nishimura DG, Pauly JM. Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE Trans Med Imaging. 2005 Jun;24(6):799-808. doi: 10.1109/TMI.2005.848376. PMID: 15959939.
        """
        return torch.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5

    @staticmethod
    def split(nufft, tile):
        split_linops = []
        for linop in nufft.linops:
            sub_tile = {dim: tile.get(dim, slice(None)) for dim in linop.dims}
            split_linops.append(type(linop).split(linop, sub_tile))
        out = copy(nufft)
        out.linops = nn.ModuleList(split_linops)
        return out

    def flatten(self):
        """Don't combine constituent linops into a chain with other linops
        Informs how split should behave
        """
        return [self]

    @property
    def device(self):
        """Tracks device of interpolating/sampling linop
        Useful for toeplitz
        """
        raise NotImplementedError()
