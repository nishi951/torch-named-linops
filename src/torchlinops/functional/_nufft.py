from torch import Tensor
from jaxtyping import Float
from types import SimpleNamespace
from math import prod

from torchlinops.linops.pad_last import PadLast, pad_to_size, crop_slice_from_pad
from torchlinops.linops.nufft import NUFFT
from torchlinops.utils import cfftn, cifftn

from ._interp.interp import interpolate, interpolate_adjoint

__all__ = ["nufft", "nufft_adjoint"]


def nufft(
    x: Tensor,
    locs: Float[Tensor, "... D"],
    oversamp: float = 1.25,
    width: float = 4.0,
):
    """Functional interface for NUFFT"""

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
    """Functional interface for adjoint NUFFT"""
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
