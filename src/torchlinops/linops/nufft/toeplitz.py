from itertools import product
from math import prod
from typing import Optional

import torch

from torchlinops.utils import cfftn, default_to

from ...nameddim import ELLIPSES
from ..dense import Dense
from ..identity import Identity
from ..namedlinop import NamedLinop
from .nufft import NUFFT
from .sampling_nufft import SamplingNUFFT
from .utils import scale_int


def toeplitz_psf(
    nufft: NUFFT,
    inner: Optional[NamedLinop] = None,
    dtype: Optional[torch.dtype] = None,
    oversamp: float = 2.0,
) -> NamedLinop:
    """Compute the Toeplitz point spread function (PSF) for a NUFFT operator.

    Constructs a PSF kernel that enables efficient ``A.H @ inner @ A``
    computation via FFT-based Toeplitz embedding, avoiding explicit
    forward/adjoint NUFFT pairs.

    Parameters
    ----------
    nufft : NUFFT
        The NUFFT operator to compute the PSF for.
    inner : NamedLinop, optional
        An optional inner linear operator applied between the forward and
        adjoint NUFFT (e.g., density compensation). If ``None``, defaults
        to the identity.
    dtype : torch.dtype, optional
        Data type for the PSF kernel. Defaults to ``torch.complex64``.
    oversamp : float, optional
        Toeplitz oversampling factor. Default is 2.0.

    Returns
    -------
    NamedLinop
        A ``Dense`` named linear operator containing the Toeplitz PSF
        kernel in the Fourier domain.
    """

    if isinstance(nufft, SamplingNUFFT):
        raise NotImplementedError(
            "Toeplitz embedding is not implemented for SamplingNUFFT"
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
    # new_locs is already scaled/shifted into the prepared interpolation
    # coordinate system, so the reconstructed NUFFT must not re-prep locs
    os_options = {**nufft.options, "skip_prep_locs": True}
    nufft_os = NUFFT(
        new_locs,
        grid_size=new_grid_size,
        output_shape=nufft.output_shape,
        input_shape=nufft.input_shape,
        input_kshape=nufft.input_kshape,
        batch_shape=nufft.batch_shape,
        **os_options,
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

    Returns
    -------
    Tensor
        The rescaled trajectory coordinates.
    """
    ndim = locs.shape[dim]
    out = []
    for d in range(ndim):
        loc = torch.select(locs, dim, d)
        # Affine transform
        loc = (loc - c0[d]) * w1[d] / w0[d] + c1[d]
        out.append(loc)
    return torch.stack(out, dim=dim)
