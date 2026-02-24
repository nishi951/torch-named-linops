from math import prod
from types import SimpleNamespace

from jaxtyping import Float
from torch import Tensor

from torchlinops.linops.nufft import NUFFT
from torchlinops.linops.pad_last import PadLast, crop_slice_from_pad, pad_to_size
from torchlinops.utils import cfftn, cifftn

from ._interp.interp import interpolate, interpolate_adjoint

__all__ = ["nufft", "nufft_adjoint"]


def nufft(
    x: Tensor,
    locs: Float[Tensor, "... D"],
    oversamp: float = 1.25,
    width: float = 4.0,
):
    """Functional interface for the Non-Uniform Fast Fourier Transform.

    Computes the forward NUFFT of input data at specified non-uniform
    locations. Internally applies apodization, zero-padding, FFT, and
    interpolation.

    Parameters
    ----------
    x : Tensor
        Input data on a regular grid. The last ``D`` dimensions are
        treated as spatial dimensions, where ``D = locs.shape[-1]``.
    locs : Float[Tensor, "... D"]
        Non-uniform sample locations. Each entry along the last dimension
        corresponds to a spatial axis and should lie in
        ``[-N//2, N//2]`` where ``N`` is the grid size along that axis.
    oversamp : float, optional
        Oversampling factor for the padded FFT grid. Default is 1.25.
    width : float, optional
        Interpolation kernel width. Default is 4.0.

    Returns
    -------
    Tensor
        NUFFT values evaluated at the non-uniform locations.
    """

    grid_size = x.shape[-locs.shape[-1] :]
    params = init_nufft(grid_size, locs, oversamp, width, x.device)

    x = x * params.apodize
    x = PadLast.fn(params.pad_ns, x)
    x = cfftn(x, dim=params.dim, norm="ortho")
    x = interpolate(
        x,
        params.locs,
        width,
        kernel="kaiser_bessel",
        kernel_params=dict(beta=params.beta),
    )
    x = x / params.scale_factor
    return x


def nufft_adjoint(
    x: Tensor,
    locs: Float[Tensor, "... D"],
    grid_size: tuple[int, ...],
    oversamp: float = 1.25,
    width: float = 4.0,
):
    """Functional interface for the adjoint NUFFT.

    Grids non-uniformly sampled data back onto a regular grid. Internally
    applies adjoint interpolation (gridding), inverse FFT, cropping, and
    apodization correction.

    Parameters
    ----------
    x : Tensor
        Non-uniformly sampled data to be gridded.
    locs : Float[Tensor, "... D"]
        Non-uniform sample locations. Each entry along the last dimension
        corresponds to a spatial axis and should lie in
        ``[-N//2, N//2]`` where ``N`` is the grid size along that axis.
    grid_size : tuple of int
        Desired output grid size for each spatial dimension.
    oversamp : float, optional
        Oversampling factor for the padded FFT grid. Default is 1.25.
    width : float, optional
        Interpolation kernel width. Default is 4.0.

    Returns
    -------
    Tensor
        Gridded data on a regular grid of shape ``(..., *grid_size)``.
    """
    params = init_nufft(grid_size, locs, oversamp, width, x.device)

    x = x / params.scale_factor
    x = interpolate_adjoint(
        x,
        params.locs,
        params.padded_size,
        width,
        kernel="kaiser_bessel",
        kernel_params=dict(beta=params.beta),
    )
    x = cifftn(x, dim=params.dim, norm="ortho")
    x = PadLast.adj_fn(params.pad_ns, x)
    x = x * params.apodize
    return x


def init_nufft(grid_size, locs, oversamp, width, device):
    """Initialize NUFFT parameters.

    Computes the oversampled grid size, interpolation kernel beta parameter,
    apodization weights, padding attributes, rescaled locations, and scaling
    factor needed by the functional NUFFT forward and adjoint passes.

    Parameters
    ----------
    grid_size : tuple of int
        Original spatial grid dimensions.
    locs : Float[Tensor, "... D"]
        Non-uniform sample locations.
    oversamp : float
        Oversampling factor for the padded FFT grid.
    width : float
        Interpolation kernel width.
    device : torch.device
        Device on which to place computed tensors.

    Returns
    -------
    SimpleNamespace
        Namespace containing the following fields:

        - ``ndim`` : number of spatial dimensions
        - ``dim`` : tuple of negative axis indices for the spatial dims
        - ``grid_size`` : original grid size
        - ``padded_size`` : oversampled grid size
        - ``locs`` : rescaled sample locations
        - ``beta`` : Kaiser-Bessel beta parameter
        - ``apodize`` : apodization correction weights
        - ``pad_ns`` : namespace with padding attributes
        - ``scale_factor`` : normalization scale factor
    """
    ndim = locs.shape[-1]
    dim = tuple(range(-ndim, 0))
    padded_size = tuple(int(s * oversamp) for s in grid_size)
    locs = NUFFT.prep_locs(locs, grid_size, padded_size)

    # Apodize weights
    beta = NUFFT.beta(width, oversamp)
    apodize = NUFFT.apodize_weights(grid_size, padded_size, oversamp, width, beta)
    apodize = apodize.to(device)

    # Pad Attrs
    pad = pad_to_size(grid_size, padded_size)
    crop_slice = crop_slice_from_pad(pad)
    pad_ns = SimpleNamespace(
        im_size=grid_size,
        pad_im_size=padded_size,
        D=ndim,
        pad=pad,
        crop_slice=crop_slice,
    )

    # Scale factor
    scale_factor = width**ndim * (prod(grid_size) / prod(padded_size)) ** 0.5
    return SimpleNamespace(
        ndim=ndim,
        dim=dim,
        grid_size=grid_size,
        padded_size=padded_size,
        locs=locs,
        beta=beta,
        apodize=apodize,
        pad_ns=pad_ns,
        scale_factor=scale_factor,
    )


# TODO: gridded_nufft, gridded_nufft_adjoint
