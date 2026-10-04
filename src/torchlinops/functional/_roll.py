"""Fused-copy roll

Helps improve performance for fftshift
"""

import torch

from itertools import product
from torch import Tensor

__all__ = ["roll", "fftshift", "ifftshift"]


def roll_fused(x: Tensor, shifts: tuple[int, ...], dims: tuple[int, ...]):
    """Fused roll that directly copies blocks to their final location.

    Drop-in replacement for torch.roll

    Parameters
    ----------
    x : Tensor
        The input tensor to roll.
    shifts : tuple[int, ...]
        Shift amounts. Mirrors torch.roll convention.
    dims : tuple[int, ...]
        Dims to shift over. Mirrors torch.roll convention.

    """
    if len(dims) != len(shifts):
        raise ValueError(
            f"Dims and shifts must have same length but got shifts: {shifts} and dims: {dims}"
        )
    if len(dims) == 0:
        return x

    slices = []
    for axis, shift in zip(dims, shifts):
        shift = shift % x.shape[axis]
        if shift != 0:
            left = (slice(None, -shift), slice(shift, None))  # (source, target)
            right = (slice(-shift, None), slice(None, shift))
        else:
            left = (slice(None), slice(None))
            right = (slice(None), slice(None))
        slices.append((left, right))

    out = torch.empty_like(x)
    for block in product(*slices):
        src_slc = [slice(None)] * x.ndim
        dst_slc = [slice(None)] * x.ndim
        for (source, target), axis in zip(block, dims):
            src_slc[axis] = source
            dst_slc[axis] = target
        out[tuple(dst_slc)].copy_(x[tuple(src_slc)])
    return out


class Roll(torch.autograd.Function):
    """Without this function, roll_fused's backward() performance suffers greatly."""

    @staticmethod
    def forward(x, shifts, dims):
        return roll_fused(x, shifts, dims)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.shifts, ctx.dims = inputs

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the shifts from the forward pass
        shifts = tuple(-s for s in ctx.shifts)
        dims = ctx.dims
        return roll_fused(grad_output, shifts, dims), None, None


def roll(
    x: Tensor,
    shifts: int | tuple[int, ...] = tuple(),
    dims: int | tuple[int, ...] = tuple(),
):
    """Convenience wrapper"""
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)
    return Roll.apply(x, shifts, dims)


def _fftshift_helper(
    x: Tensor, dim: int | tuple[int, ...] | None = None, inverse=False
):
    """fftshift helper

    Mirrors torch.fft.fftshift.

    Parameters
    ----------
    x : Tensor
        Input tensor to shift.
    dim : int or tuple of ints or None
        Dims to fftshift over.
    inverse : bool, default False
        If True, perform inverse fftshift (ifftshift) instead.


    Notes
    -----
    Worked example:

    x.shape = (4, 7)
    case inverse = False:
        shifts should be (+2, +3)
    case inverse = True
        shifts should be (+2, +4)
    (4 + 1) // 2 = 2
    (7 + 1) // 2 = 4


    """
    if isinstance(dim, int):
        dims = (dim,)
    elif dim is None:
        dims = tuple(range(x.ndim))
    else:
        dims = dim

    assert isinstance(dims, tuple)
    shifts = tuple((x.shape[ax] + (1 if inverse else 0)) // 2 for ax in dims)
    return roll(x, shifts, dims)


def fftshift(x, dim: int | tuple[int, ...] | None = None):
    return _fftshift_helper(x, dim, inverse=False)


def ifftshift(x, dim: int | tuple[int, ...] | None = None):
    return _fftshift_helper(x, dim, inverse=True)
