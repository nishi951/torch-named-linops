from collections.abc import Callable
from copy import copy, deepcopy
from itertools import product
from math import prod
from typing import Literal, Optional
from warnings import warn

import torch
import torch.nn as nn
from jaxtyping import Float, Shaped
from torch import Tensor

from torchlinops.utils import cfftn, default_to
from .chain import Chain
from .dense import Dense
from .diagonal import Diagonal
from .fft import FFT
from .identity import Identity
from .interp import Interpolate
from .nameddim import ELLIPSES, ND, NS, Shape, get_nd_shape
from .namedlinop import NamedLinop
from .pad_last import PadLast
from .sampling import Sampling
from .scalar import Scalar

__all__ = ["NUFFT"]


class NUFFT(Chain):
    def __init__(
        self,
        locs: Float[Tensor, "... D"],
        grid_size: tuple[int, ...],
        output_shape: Shape,
        input_shape: Optional[Shape] = None,
        input_kshape: Optional[Shape] = None,
        batch_shape: Optional[Shape] = None,
        oversamp: float = 1.25,
        width: float = 4.0,
        mode: Literal["interpolate", "sampling"] = "interpolate",
        do_prep_locs: bool = True,
        apodize_weights: Optional[Float[Tensor, "..."]] = None,
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

        oversamp : float
            Oversampling factor for fourier domain grid
        width : float
            Width of kernel to use for interpolation
        mode : str, "interpolate" or "sampling"
        do_prep_locs : bool, default True
            Whether to scale, shift, and clamp the locs to be amenable to interpolation
            By default (=True), assumes the locs lie in [-N/2, N/2]
                Scales, shifts and clamps them them to [0, oversamp*N - 1]
            If False, does not do this, which can have some benefits for memory reasons
        apodize_weights : Optional[Tensor]
            Provide apodization weights
            Only relevant for "intepolate" mode
            Can have memory benefits
        **options : dict
            Additional options
            toeplitz : bool
                If True, normal() performs toeplitz embedding calculation
            toeplitz_dtype : torch.dtype
                Data type for the toeplitz embedding. Probably should be torch.complex64

        """
        device = locs.device
        self.mode = mode
        self.options = options
        # Infer shapes
        self.input_shape = ND.infer(default_to(get_nd_shape(grid_size), input_shape))
        self.input_kshape = ND.infer(
            default_to(get_nd_shape(grid_size, kspace=True), input_kshape)
        )
        self.output_shape = ND.infer(output_shape)
        self.batch_shape = ND.infer(default_to(("...",), batch_shape))
        batched_input_shape = NS(batch_shape) + NS(self.input_shape)

        # Initialize variables
        ndim = len(grid_size)
        padded_size = [int(i * oversamp) for i in grid_size]

        # Create Padding
        pad = PadLast(
            padded_size,
            grid_size,
            in_shape=self.input_shape,
            batch_shape=self.batch_shape,
        )

        # Create FFT
        fft = FFT(
            ndim=locs.shape[-1],
            centered=True,
            norm="ortho",
            batch_shape=self.batch_shape,
            grid_shapes=(pad.out_im_shape, self.input_kshape),
        )

        # Create Interpolator
        grid_shape = fft._shape.output_grid_shape
        if do_prep_locs:
            locs_prepared = self.prep_locs(
                locs,
                grid_size,
                padded_size,
                nufft_mode=mode,
            )
        else:
            locs_prepared = locs
        if self.mode == "interpolate":
            beta = self.beta(width, oversamp)
            # Create Apodization
            if apodize_weights is None:
                weight = self.apodize_weights(
                    grid_size, padded_size, oversamp, width, beta
                ).to(device)  # Helps with batching later
            else:
                weight = apodize_weights
            if weight.isnan().any() or weight.isinf().any():
                raise ValueError(
                    f"Nan/Inf values detected in apodization weight (width={width}, oversamp={oversamp})."
                )
            apodize = Diagonal(weight, batched_input_shape.ishape)
            apodize.name = "Apodize"

            # Create Interpolator
            interp = Interpolate(
                locs_prepared,
                padded_size,
                batch_shape=self.batch_shape,
                locs_batch_shape=self.output_shape,
                grid_shape=grid_shape,
                width=width,
                kernel="kaiser_bessel",
                kernel_params=dict(beta=beta),
            )
            # Create scaling
            scale_factor = width**ndim * (prod(grid_size) / prod(padded_size)) ** 0.5
            scale = Scalar(weight=1.0 / scale_factor, ioshape=interp.oshape)
            scale.to(device)  # Helps with batching later
            linops = [apodize, pad, fft, interp, scale]
        elif self.mode == "sampling":
            if locs_prepared.is_complex() or locs_prepared.is_floating_point():
                raise ValueError(
                    f"Sampling linop requries integer-type locs but got {locs_prepared.dtype}"
                )
            # Clamp to within range
            interp = Sampling.from_stacked_idx(
                locs_prepared,
                dim=-1,
                # Arguments for Sampling
                input_size=padded_size,
                output_shape=self.output_shape,
                input_shape=grid_shape,
                batch_shape=self.batch_shape,
            )
            # No apodization or scaling needed
            linops = [pad, fft, interp]
        else:
            raise ValueError(f"Unrecognized NUFFT mode: {mode}")

        super().__init__(*linops, name="NUFFT")
        # Useful parameters to save
        self.locs = locs
        self.grid_size = grid_size
        self.oversamp = oversamp
        self.width = width

        # Handles to get modules directly
        self.pad = pad
        self.fft = fft
        self.interp = interp

    def adjoint(self):
        # Hybrid of chain adjoint and namedlinop adjoint
        adj = copy(self)
        adj._shape = adj._shape.H

        linops = list(linop.adjoint() for linop in reversed(self.linops))
        adj.linops = nn.ModuleList(linops)
        return adj

    def normal(self, inner=None):
        if self.options.get("toeplitz", False):
            dtype = self.options.get("toeplitz_dtype")
            oversamp = self.options.get("toeplitz_oversamp", 2.0)
            toep_kernel = toeplitz_psf(self, inner, dtype=dtype, oversamp=oversamp)
            pad = PadLast(
                scale_int(self.grid_size, oversamp),
                self.grid_size,
                in_shape=self.input_shape,
                batch_shape=self.batch_shape,
            )
            fft = self.fft
            return pad.normal(fft.normal(toep_kernel))
        return super().normal(inner)

    @staticmethod
    def prep_locs(
        locs: Shaped[Tensor, "... D"],
        grid_size: tuple,
        padded_size: tuple,
        pad_mode: Literal["zero", "circular"] = "circular",
        nufft_mode: Literal["interpolate", "sampling"] = "interpolate",
    ):
        """
        Parameters
        ----------
        locs : Shaped[Tensor, "... D"]
            Input tensor representing locations in the grid. The last dimension corresponds to spatial dimensions.
            Range is [-N//2, N//2]
        grid_size : tuple
            The original size of the grid before padding.
        padded_size : tuple
            The size of the grid after padding.
        pad_mode : Literal["zero", "circular"], optional
            The type of padding applied. Can be "zero" for zero-padding or "circular" for circular padding.
            Default is "circular".
        nufft_mode : Literal["interpolate", "sampling"], optional
            The mode of the NUFFT operation. Can be "interpolate" for interpolation or "sampling" for sampling.
            Default is "interpolate".

        Returns
        -------
        Shaped[Tensor, "... D"]
            Adjusted locations tensor based on the specified padding and NUFFT modes.
            Range is [0, N_pad].
            dtype is floating-point if nufft_mode is "interpolate", and integer
            if nufft_mode is "sampling"

        Raises
        ------
        ValueError
            If an unrecognized `pad_mode` is provided.

        Examples
        --------
        >>> _ = torch.manual_seed(0);
        >>> locs = torch.rand(1000, 3) * 64 - 32 # [-32, 32]
        >>> locs.min()
        tensor(-31.9949)
        >>> locs.max()
        tensor(31.9896)
        >>> grid_size = (64, 64, 64)
        >>> padded_size = (80, 80, 80) # oversamp = 1.25
        >>> locs_scaled_shifted = NUFFT.prep_locs(locs, grid_size, padded_size)
        >>> locs_scaled_shifted.min()
        tensor(0.0064)
        >>> locs_scaled_shifted.max()
        tensor(79.9871)

        >>> _ = torch.manual_seed(0);
        >>> locs = torch.rand(1000, 3) * 64 - 32 # [-32, 32]
        >>> locs = torch.round(locs * 1.25) / 1.25
        >>> grid_size = (64, 64, 64)
        >>> padded_size = (80, 80, 80) # oversamp = 1.25
        >>> locs_scaled_shifted = NUFFT.prep_locs(locs, grid_size, padded_size, nufft_mode='sampling')
        >>> locs_scaled_shifted.min()
        tensor(0)
        >>> locs_scaled_shifted.max()
        tensor(79)



        Notes
        -----
        - Assumes that the input `locs` are centered.
        - Adjusts the locations by scaling and shifting them according to the grid and padded sizes.
        - Applies clamping or remainder operations based on the padding mode and NUFFT mode.
        """
        # Clone to prevent in-place scaling from modifying the original
        out = locs.clone()
        for i in range(-len(grid_size), 0):
            out[..., i] *= padded_size[i] / grid_size[i]
            out[..., i] += padded_size[i] // 2
            if pad_mode == "zero":
                out[..., i] = torch.clamp(out[..., i], 0, padded_size[i] - 1)
            elif pad_mode == "circular":
                if nufft_mode == "interpolate":
                    out[..., i] = torch.remainder(
                        out[..., i], torch.tensor(padded_size[i])
                    )
                elif nufft_mode == "sampling":
                    # Wrap rounded index to other side of kspace
                    out[..., i] = torch.round(out[..., i])
                    out[..., i] = torch.remainder(out[..., i], padded_size[i])
            else:
                raise ValueError(f"Unrecognized padding mode during prep: {pad_mode}")
        if nufft_mode == "sampling":
            out = out.to(torch.int64)
        return out

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
    def apodize_weights(grid_size, padded_size, oversamp, width: float, beta: float):
        grid_size = torch.tensor(grid_size)
        padded_size = torch.tensor(padded_size)
        grid = torch.meshgrid(*(torch.arange(s) for s in grid_size), indexing="ij")
        grid = torch.stack(grid, dim=-1)

        # Sigpy compatibility
        apod = (
            beta**2 - (torch.pi * width * (grid - grid_size // 2) / padded_size) ** 2
        ) ** 0.5
        apod /= torch.sinh(apod)

        # Beatty paper
        # apod = (torch.pi * width * (grid - grid_size // 2) / padded_size) ** 2 - beta**2
        # print(apod)
        apod = torch.prod(apod, dim=-1)
        return apod

    def split_forward(self, ibatch, obatch):
        ibatch_lookup = {d: slc for d, slc in zip(self.ishape, ibatch)}
        obatch_lookup = {d: slc for d, slc in zip(self.oshape, obatch)}
        split_linops = []
        for linop in self.linops:
            sub_ibatch = [ibatch_lookup.get(dim, slice(None)) for dim in linop.ishape]
            sub_obatch = [obatch_lookup.get(dim, slice(None)) for dim in linop.oshape]
            split_linops.append(linop.split_forward(sub_ibatch, sub_obatch))
        out = copy(self)
        out.linops = nn.ModuleList(split_linops)
        return out

    def flatten(self):
        """Don't combine constituent linops into a chain with other linops
        Informs how split_forward should behave
        """
        return [self]

    @property
    def device(self):
        """Tracks device of interpolating/sampling linop
        Useful for toeplitz
        """
        if self.mode == "interpolate":
            return self.interp.locs.device
        elif self.mode == "sampling":
            return self.interp.idx[0].device
        raise ValueError(f"Unrecognized NUFFT mode: {self.mode}")


def toeplitz_psf(
    nufft,
    inner: Optional[NamedLinop] = None,
    dtype: Optional[torch.dtype] = None,
    oversamp: float = 2.0,
) -> NamedLinop:
    """Compute the toeplitz PSF for this NUFFT, with
    # TODO: maybe accommodate other oversampling factors (more complicated)
    """
    if isinstance(nufft.interp, Sampling):
        raise NotImplementedError(
            f"Toeplitz embedding not yet implemented for Sampling-type NUFFT"
        )

    # Initialize variables
    dtype = default_to(torch.complex64, dtype)
    device = nufft.device
    grid_size = nufft.pad.im_size
    new_grid_size = scale_int(grid_size, oversamp)
    ndim = len(grid_size)
    width = nufft.pad.pad_im_size
    new_width = scale_int(width, oversamp)
    new_locs = rescale_locs(
        nufft.interp.locs.clone(),
        c0=tuple(w // 2 for w in width),
        w0=width,
        c1=tuple(w // 2 for w in new_width),
        w1=new_width,
    )
    nufft_os = NUFFT(
        new_locs,
        grid_size=new_grid_size,
        output_shape=nufft.output_shape,
        input_shape=nufft.input_shape,
        input_kshape=nufft.input_kshape,
        batch_shape=nufft.batch_shape,
        oversamp=nufft.oversamp,  # Oversample on top of toeplitz oversampling
        width=nufft.width,
        mode=nufft.mode,
        do_prep_locs=False,
    )

    # Initialize inner if not provided
    if inner is None:
        inner = Identity(ishape=nufft.oshape)

    if len(inner.ishape) != len(inner.oshape):
        raise ValueError(
            f"Inner linop must have identical input and output shape lengths but got ishape={inner.ishape} and oshape={inner.oshape}"
        )

    # Get all useful shapes and sizes
    kernel_shape, ishape, oshape, kernel_size, input_size, batch_sizes = psf_sizing(
        nufft, inner, oversamp
    )

    # Create empty kernel
    kernel = torch.zeros(*kernel_size, dtype=dtype, device=device)

    # Allocate input
    allones = torch.zeros(*input_size, dtype=dtype, device=device)
    scale_factor = oversamp**ndim / (prod(new_grid_size) ** 0.5)

    # Compute kernel by iterating through all possible input-output pairs
    dim = tuple(range(-len(new_grid_size), 0))
    for batch_idx in all_indices(batch_sizes):
        allones[batch_idx] = 1.0
        otf = nufft_os.H(inner(allones))
        kernel[batch_idx] = cfftn(otf, dim=dim, norm=None) * scale_factor
        allones[batch_idx] = 0.0  # reset
    kernel_os = Dense(
        weight=kernel,
        weightshape=kernel_shape,
        ishape=ishape,
        oshape=oshape,
    )

    return kernel_os


def scale_int(t: tuple[int, ...], scale_factor: float):
    return tuple(int(scale_factor * s) for s in t)


def psf_sizing(nufft, inner: NamedLinop, toeplitz_oversamp: float = 2.0):
    """Helper function for computing shapes and sizes of kernels and inputs"""
    n_output_dims = len(nufft.output_shape)

    # Get all relevant shapes
    batch_ishape, batch_oshape = (
        inner.ishape[:-n_output_dims],
        inner.oshape[:-n_output_dims],
    )
    io_kshape = nufft.fft._shape.output_grid_shape
    ishape = batch_ishape + io_kshape
    oshape = batch_oshape + io_kshape
    if batch_ishape == (ELLIPSES,):  # Special case
        kernel_shape = batch_ishape + io_kshape
    elif ELLIPSES in batch_ishape and ELLIPSES in batch_oshape:
        raise ValueError(
            f"Underspecified kernel shape for toeplitz embedding with inner.shape = {inner.shape}. Specify more dimensions of inner to avoid this."
        )
    else:
        kernel_shape = batch_ishape + batch_oshape + io_kshape

    # Get batch sizes
    batch_sizes = tuple(inner.size(d) for d in batch_ishape)
    batch_sizes = tuple(a if a is not None else 1 for a in batch_sizes)

    # Get kernel size
    im_size = nufft.pad.im_size
    kernel_ksize = scale_int(im_size, toeplitz_oversamp)
    kernel_size = batch_sizes + batch_sizes + kernel_ksize

    # Get test input size
    output_size = tuple(nufft.size(d) for d in nufft.output_shape)
    input_size = batch_sizes + output_size

    return kernel_shape, ishape, oshape, kernel_size, input_size, batch_sizes


def all_indices(size: tuple[int]):
    ranges = tuple(range(s) for s in size)
    return product(*ranges)


def rescale_locs(locs, c0: tuple, w0: tuple, c1: tuple, w1: tuple, dim: int = -1):
    """Perform a scale-and-shift operation on a single dimension of a locs tensor.
    Parameters
    ----------
    locs : Tensor
        The locs to rescale, shape [... D ...]
    c0, w0 : tuple
        The center and width parameter for the current locs
    c1, w1: tuple
        The desired center and width parameters.
    dim : int
        The dimension of locs to unstack
    """
    ndim = locs.shape[dim]
    out = []
    for d in range(ndim):
        loc = torch.select(locs, dim, d)
        # Affine transform
        loc = (loc - c0[d]) * w1[d] / w0[d] + c1[d]
        out.append(loc)
    return torch.stack(out, dim=dim)
