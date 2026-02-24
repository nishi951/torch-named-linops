"""Differentiable implementations of Grid/Ungrid"""

from jaxtyping import Float, Inexact
from torch import Tensor
from torch.autograd import Function

from .grid import grid
from .ungrid import ungrid

__all__ = ["interpolate", "interpolate_adjoint"]


class InterpolateFn(Function):
    """Equal to ungrid"""

    @staticmethod
    def forward(
        vals: Inexact[Tensor, "..."],
        locs: Float[Tensor, "... D"],
        width: float | tuple[float, ...],
        kernel: str,
        norm: int,
        pad_mode: str,
        kernel_params: dict,
    ) -> Tensor:
        output = ungrid(vals, locs, width, kernel, norm, pad_mode, kernel_params)
        output.requires_grad_(vals.requires_grad)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Unpack input and output
        vals, locs, width, kernel, norm, pad_mode, kernel_params = inputs

        # Save for backward pass
        ndim = locs.shape[-1]
        ctx.grid_size = vals.shape[-ndim:]
        ctx.width = width
        ctx.kernel = kernel
        ctx.norm = norm
        ctx.pad_mode = pad_mode
        ctx.kernel_params = kernel_params
        ctx.save_for_backward(locs)

    @staticmethod
    def backward(ctx, grad_output):
        grad_vals = grad_locs = grad_width = grad_kernel = grad_norm = grad_pad_mode = (
            grad_kernel_params
        ) = None
        if ctx.needs_input_grad[0]:
            locs = ctx.saved_tensors[0]
            grad_vals = grid(
                grad_output,
                locs,
                ctx.grid_size,
                ctx.width,
                ctx.kernel,
                ctx.norm,
                ctx.pad_mode,
                ctx.kernel_params,
            )
        return (
            grad_vals,
            grad_locs,
            grad_width,
            grad_kernel,
            grad_norm,
            grad_pad_mode,
            grad_kernel_params,
        )


def interpolate(
    vals: Inexact[Tensor, "..."],
    locs: Float[Tensor, "... D"],
    width: float | tuple[float, ...],
    kernel="kaiser_bessel",
    norm: int = 1,
    pad_mode: str = "circular",
    kernel_params: dict = None,
):
    """Interpolate from a regular grid to scattered locations (ungridding).

    Evaluates values on a uniform grid at arbitrary non-uniform locations
    using kernel-based interpolation. This is the forward NUFFT interpolation
    step. Gradients are computed via the adjoint (gridding) operation.

    Parameters
    ----------
    vals : Inexact[Tensor, "..."]
        Values on a regular grid. The last ``D`` dimensions are spatial,
        where ``D = locs.shape[-1]``.
    locs : Float[Tensor, "... D"]
        Non-uniform target locations, with coordinates in the range
        ``[0, N-1]`` for each spatial dimension of size ``N``.
    width : float or tuple of float
        Interpolation kernel width (in grid units) for each spatial
        dimension. A scalar applies the same width to all dimensions.
    kernel : str, optional
        Interpolation kernel type. Default is ``'kaiser_bessel'``.
    norm : int, optional
        Kernel normalization order. Default is 1.
    pad_mode : str, optional
        Padding mode for out-of-bounds access. Default is ``'circular'``.
    kernel_params : dict, optional
        Additional parameters passed to the interpolation kernel
        (e.g., ``{'beta': ...}`` for Kaiser-Bessel).

    Returns
    -------
    Tensor
        Interpolated values at the non-uniform locations.
    """
    return InterpolateFn.apply(vals, locs, width, kernel, norm, pad_mode, kernel_params)


class InterpolateAdjointFn(Function):
    """Equal to grid."""

    @staticmethod
    def forward(
        vals: Inexact[Tensor, "..."],
        locs: Float[Tensor, "... D"],
        grid_size: tuple[int, ...],
        width: float | tuple[float, ...],
        kernel: str,
        norm: int,
        pad_mode: str,
        kernel_params: dict,
    ) -> Tensor:
        output = grid(
            vals,
            locs,
            grid_size,
            width,
            kernel,
            norm,
            pad_mode,
            kernel_params,
        )
        output.requires_grad_(vals.requires_grad)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        vals, locs, grid_size, width, kernel, norm, pad_mode, kernel_params = inputs

        # Save for backward pass
        ctx.width = width
        ctx.kernel = kernel
        ctx.norm = norm
        ctx.pad_mode = pad_mode
        ctx.kernel_params = kernel_params
        ctx.save_for_backward(locs)

    @staticmethod
    def backward(ctx, grad_output):
        grad_vals = grad_locs = grad_grid_size = grad_width = grad_kernel = (
            grad_norm
        ) = grad_pad_mode = grad_kernel_params = None
        if ctx.needs_input_grad[0]:
            locs = ctx.saved_tensors[0]
            grad_vals = ungrid(
                grad_output,
                locs,
                ctx.width,
                ctx.kernel,
                ctx.norm,
                ctx.pad_mode,
                ctx.kernel_params,
            )
        return (
            grad_vals,
            grad_locs,
            grad_grid_size,
            grad_width,
            grad_kernel,
            grad_norm,
            grad_pad_mode,
            grad_kernel_params,
        )


def interpolate_adjoint(
    vals: Inexact[Tensor, "..."],
    locs: Float[Tensor, "... D"],
    grid_size: tuple[int, ...],
    width: float | tuple[float, ...],
    kernel: str = "kaiser_bessel",
    norm: int = 1,
    pad_mode: str = "circular",
    kernel_params: dict = None,
):
    """Adjoint of interpolation (gridding) from scattered locations to a regular grid.

    Scatters values from non-uniform locations back onto a regular grid
    using kernel-based gridding. This is the adjoint of the ``interpolate``
    operation and corresponds to the gridding step in an adjoint NUFFT.

    Parameters
    ----------
    vals : Inexact[Tensor, "..."]
        Values at non-uniform locations to be gridded.
    locs : Float[Tensor, "... D"]
        Non-uniform source locations, with coordinates in the range
        ``[0, N-1]`` for each spatial dimension of size ``N``.
    grid_size : tuple of int
        Output grid size for each spatial dimension.
    width : float or tuple of float
        Interpolation kernel width (in grid units) for each spatial
        dimension. A scalar applies the same width to all dimensions.
    kernel : str, optional
        Interpolation kernel type. Default is ``'kaiser_bessel'``.
    norm : int, optional
        Kernel normalization order. Default is 1.
    pad_mode : str, optional
        Padding mode for out-of-bounds access. Default is ``'circular'``.
    kernel_params : dict, optional
        Additional parameters passed to the interpolation kernel
        (e.g., ``{'beta': ...}`` for Kaiser-Bessel).

    Returns
    -------
    Tensor
        Gridded values on a regular grid of shape ``(..., *grid_size)``.
    """
    return InterpolateAdjointFn.apply(
        vals,
        locs,
        grid_size,
        width,
        kernel,
        norm,
        pad_mode,
        kernel_params,
    )
