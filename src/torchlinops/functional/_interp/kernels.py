from jaxtyping import Float, Integer, Bool
from torch import Tensor
from typing import Literal
from collections.abc import Callable

from functools import partial

import torch
import triton
import triton.language as tl

__all__ = [
    "kaiser_bessel",
    "kaiser_bessel_torch",
    "spline_torch",
    "spline",
]


@torch.compile
def kaiser_bessel_torch(x: Float[Tensor, "..."], beta: float):
    """Vectorized kaiser-bessel kernel implementation
    Parameters
    ----------
    x: Tensor, any shape
        Kernel acts on favlues in x in the range [-1, 1]
    beta: float
        Shape parameter for kaiser bessel kernel

    Explanation
    -----------
    The scary polynomial stuff is a polynomial approximation of the zeroth-order modified Bessel
    function of the first kind.
    Here, beta is standing in for a bunch of constant factors that multiply in front of it.

    References
    ----------
    Polynomial approximation of Bessel function from
    Abramowitz and Stegun, Handbook of Mathematical Functions, p.378
    https://personal.math.ubc.ca/~cbm/aands/page_378.htm
    """
    x = beta * torch.sqrt(1 - x**2)
    smallmask = x < 3.75
    t = x / 3.75
    # Optimized computation ordering
    t2 = t * t
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    t10 = t6 * t4
    t12 = t6 * t6

    t3 = t2 * t
    t5 = t3 * t2
    t7 = t5 * t2

    small = (
        1
        + 3.5156229 * t2
        + 3.0899424 * t4
        + 1.2067492 * t6
        + 0.2659732 * t8
        + 0.0360768 * t10
        + 0.0045813 * t12
    )
    big = (
        torch.exp(x)
        / torch.sqrt(x)
        * (
            0.39894228
            + 0.01328592 / t
            + 0.00225319 / t2
            - 0.00157565 / t3
            + 0.00916281 / t4
            - 0.02057706 / t5
            + 0.02635537 / t6
            - 0.01647633 / t7
            + 0.00392377 / t8
        )
    )
    return torch.where(smallmask, small, big)


@torch.compile
def spline_torch(x):
    return torch.maximum(1.0 - torch.abs(x), torch.tensor(0.0))


def get_kernel_fn(kernel, kernel_params) -> Callable[[Tensor], Tensor]:
    if kernel == "spline":
        kernel_fn = spline_torch
    elif kernel == "kaiser_bessel":
        beta = kernel_params.get("beta", 1.0)
        kernel_fn = partial(kaiser_bessel_torch, beta=beta)
    else:
        raise ValueError(f"Unrecognized kernel {kernel}")
    return kernel_fn


def weights_torch(
    locs: Float[Tensor, "P G D"],
    grid_locs: Float[Tensor, "P G D"],
    radius: Float[Tensor, "D"],  # noqa: F821
    norm: Literal["1", "2"],
    kernel_fn: Callable,
    grid_size: Integer[Tensor, "D"],  # noqa: F821
    padding: Literal["zero", "circular"],
) -> tuple[Float[Tensor, "P G"], Integer[Tensor, "P G D"], Bool[Tensor, "P G"]]:
    """
    P: Number of pts
    G: Patch size (flattened)
    D: Dim (e.g. 3D)

    norm : str, 1 or 2
        Type of norm to use
    """
    # Normalized delta(locations)
    dlocs = (locs[:, None] - grid_locs) / radius

    # Weights shape: [npts, npatch]
    # Mask shape: [npts, npatch]
    if norm == "2":
        dist = torch.linalg.vector_norm(dlocs, dim=-1)
        weights = kernel_fn(dist)
        mask = dist <= radius
    elif norm == "1":
        weights = kernel_fn(dlocs).prod(dim=-1)
        mask = (torch.abs(dlocs) <= radius).all(dim=-1)
    else:
        raise ValueError(f"Unexpected norm: {norm}.")
    weights[~weights.isfinite()] = 0.0

    # Edge behavior
    if padding == "zero":
        mask &= (grid_locs >= 0).all(dim=-1) & (grid_locs < grid_size).all(dim=-1)
    # This does circular padding! unless the mask above is active
    grid_locs = torch.remainder(grid_locs, grid_size)
    grid_locs = grid_locs.to(torch.int64)
    return weights, grid_locs, mask


# Euclidean norm
@triton.jit
def norm1d(vx):
    return tl.abs(vx)


@triton.jit
def norm2d(vx, vy):
    absvx = tl.abs(vx)
    absvy = tl.abs(vy)
    return tl.sqrt(absvx * absvx + absvy * absvy)


@triton.jit
def norm3d(vx, vy, vz):
    absvx = tl.abs(vx)
    absvy = tl.abs(vy)
    absvz = tl.abs(vz)
    return tl.sqrt(absvx * absvx + absvy * absvy + absvz * absvz)


@triton.jit
def kaiser_bessel(x, beta):
    """Vectorized kaiser-bessel kernel implementation
    Parameters
    ----------
    x: Tensor, any shape
        Kernel acts on favlues in x in the range [-1, 1]
    beta: float
        Shape parameter for kaiser bessel kernel
        1.0 is a good default
        Ultimately, should depend on NUFFT width/oversampling

    Explanation
    -----------
    The scary polynomial stuff is a polynomial approximation of the zeroth-order modified Bessel
    function of the first kind.
    Here, beta is standing in for a bunch of constant factors that multiply in front of it.
    If using a NUFFT, beta can be determined via the formula

    `beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5`

    References
    ----------
    Polynomial approximation of Bessel function from
    Abramowitz and Stegun, Handbook of Mathematical Functions, p.378
    https://personal.math.ubc.ca/~cbm/aands/page_378.htm
    """
    x = beta * tl.sqrt(1 - x * x)
    smallmask = x < 3.75
    t = x / 3.75
    # Optimized computation ordering
    t2 = t * t
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    t10 = t6 * t4
    t12 = t6 * t6

    t3 = t2 * t
    t5 = t3 * t2
    t7 = t5 * t2

    small = (
        1
        + 3.5156229 * t2
        + 3.0899424 * t4
        + 1.2067492 * t6
        + 0.2659732 * t8
        + 0.0360768 * t10
        + 0.0045813 * t12
    )
    big = (
        tl.exp(x)
        / tl.sqrt(x)
        * (
            0.39894228
            + 0.01328592 / t
            + 0.00225319 / t2
            - 0.00157565 / t3
            + 0.00916281 / t4
            - 0.02057706 / t5
            + 0.02635537 / t6
            - 0.01647633 / t7
            + 0.00392377 / t8
        )
    )
    return tl.where(smallmask, small, big)


@triton.jit
def spline(x):
    return tl.maximum(1.0 - tl.abs(x), 0.0)


### Common functions for grid/ungrid
KernelTypeStr = Literal["kaiser_bessel", "spline"]


def _apply_default_kernel_params(kernel: KernelTypeStr, kernel_params: dict):
    if kernel == "kaiser_bessel":
        kernel_params["beta"] = kernel_params.get("beta", 1.0)
    elif kernel == "spline":
        pass
    else:
        raise ValueError(f"Unrecognized kernel type: {kernel}")
    return kernel_params


@triton.jit
def weights1d(
    x_target,
    x_kernel_width,  # width to load
    x_base_range,  # block array to use to load
    KERNEL: tl.constexpr,  # Kernel type
    beta,
):
    # Compute neighborhood of target point
    x_range = get_neighborhood(x_target, x_kernel_width, x_base_range)

    # Compute kernel weights
    dx, rx = x_range - x_target, x_kernel_width / 2.0
    x_mask = tl.abs(dx) <= rx
    d = norm1d(dx) / rx
    if KERNEL == "kaiser_bessel":
        weights = kaiser_bessel(d, beta)
    elif KERNEL == "spline":
        weights = spline(d)
    else:
        tl.device_assert(False, f"Invalid kernel type: {KERNEL}")
    # Mask off out-of-range points
    weights_mask = x_mask
    weights = tl.where(weights_mask, weights, 0.0)
    return weights, x_range, x_mask


@triton.jit
def weights2d(
    x_target,
    y_target,
    x_kernel_width,
    y_kernel_width,
    x_base_range,
    y_base_range,
    KERNEL: tl.constexpr,
    NORM: tl.constexpr,
    beta,
):
    # Compute neighborhood of target point
    x_range = get_neighborhood(x_target, x_kernel_width, x_base_range)
    y_range = get_neighborhood(y_target, y_kernel_width, y_base_range)

    # Compute kernel weights
    dx, rx = (x_range - x_target), (x_kernel_width / 2.0)
    x_mask = tl.abs(dx) <= rx
    dy, ry = (y_range - y_target), (y_kernel_width / 2.0)
    y_mask = tl.abs(dy) <= ry
    if NORM == "2":
        d = norm2d(dx / rx, dy / ry)
        if KERNEL == "kaiser_bessel":
            weights = kaiser_bessel(d, beta)
        elif KERNEL == "spline":
            weights = spline(d)
        else:
            tl.device_assert(False, f"Invalid kernel type: {KERNEL}")
    elif NORM == "1":
        if KERNEL == "kaiser_bessel":
            wx = kaiser_bessel(norm1d(dx) / rx, beta)
            wy = kaiser_bessel(norm1d(dy) / ry, beta)
            weights = wx[:, None] * wy[None, :]
        elif KERNEL == "spline":
            wx = spline(norm1d(dx) / rx)
            wy = spline(norm1d(dy) / ry)
            weights = wx[:, None] * wy[None, :]
        else:
            tl.device_assert(False, f"Invalid kernel type: {KERNEL}")
    # Mask off NaNs and out-of-range values
    weights_mask = x_mask[:, None] & y_mask[None, :]
    weights = tl.where(weights_mask, weights, 0.0)
    return weights, x_range, y_range, x_mask, y_mask


@triton.jit
def weights3d(
    x_target,
    y_target,
    z_target,
    x_kernel_width,
    y_kernel_width,
    z_kernel_width,
    x_base_range,
    y_base_range,
    z_base_range,
    KERNEL: tl.constexpr,
    NORM: tl.constexpr,
    beta,
):
    # Compute neighborhood of target point
    x_range = get_neighborhood(x_target, x_kernel_width, x_base_range)
    y_range = get_neighborhood(y_target, y_kernel_width, y_base_range)
    z_range = get_neighborhood(z_target, z_kernel_width, z_base_range)

    # Compute kernel weights
    dx, rx = (x_range - x_target), (x_kernel_width / 2.0)
    x_mask = tl.abs(dx) <= rx
    dy, ry = (y_range - y_target), (y_kernel_width / 2.0)
    y_mask = tl.abs(dy) <= ry
    dz, rz = (z_range - z_target), (z_kernel_width / 2.0)
    z_mask = tl.abs(dz) <= rz
    if NORM == "2":
        d = norm3d(dx / rx, dy / ry, dz / rz)
        if KERNEL == "kaiser_bessel":
            weights = kaiser_bessel(d, beta)
        elif KERNEL == "spline":
            weights = spline(d)
        else:
            tl.device_assert(False, f"Invalid kernel type: {KERNEL}")
    elif NORM == "1":
        if KERNEL == "kaiser_bessel":
            wx = kaiser_bessel(norm1d(dx) / rx, beta)
            wy = kaiser_bessel(norm1d(dy) / ry, beta)
            wz = kaiser_bessel(norm1d(dz) / rz, beta)
            weights = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        elif KERNEL == "spline":
            wx = spline(norm1d(dx) / rx)
            wy = spline(norm1d(dy) / ry)
            wz = spline(norm1d(dz) / rz)
            weights = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        else:
            tl.device_assert(False, f"Invalid kernel type: {KERNEL}")
    weights_mask = x_mask[:, None, None] & (
        y_mask[None, :, None] & z_mask[None, None, :]
    )
    weights = tl.where(weights_mask, weights, 0.0)
    return weights, x_range, y_range, z_range, x_mask, y_mask, z_mask


@triton.jit
def get_neighborhood(target, kernel_width, base_range):
    lower = target - (kernel_width / 2.0)
    lower = tl.ceil(lower)
    lower = tl.cast(lower, tl.int32)
    return base_range + lower


@triton.jit
def mod_pos(t, n):
    """Modulo but ensures positive return value"""
    return tl.where(t >= 0, t % n, (t % n) + n)
