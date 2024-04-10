"""(cu)fiNUFFT backend for Autograd-Compatible NUFFT

Does not support differentiation w.r.t. coordinates

Standalone file (no NamedLinop)

"""

from math import prod, sqrt
from typing import Tuple, Optional

import finufft as F
import cufinufft as cF

import torch
from torch.autograd import Function

__all__ = ["nufft", "nufft_adjoint"]

# lookup[(<device>, <im_dim>)] -> (forward, adjoint)
# Type-2 NUFFT = "forward" NUFFT (uniform to nonuniform)
# Type-1 NUFFT = "adjoint" NUFFT (nonuniform to uniform)
get_nufft = {
    "cpu": {
        2: (F.nufft2d2, F.nufft2d1),
        3: (F.nufft3d2, F.nufft3d1),
    },
    "gpu": {
        2: (cF.nufft2d2, cF.nufft2d1),
        3: (cF.nufft3d2, cF.nufft3d1),
    },
}


def coord2contig(coord, dim=-1):
    return tuple(
        torch.select(coord, dim, i).contiguous() for i in range(coord.shape[dim])
    )


def slicelen(n, start=None, end=None, step=None):
    return len(range(*slice(start, end, step).indices(n)))


def flatten(x, start_dim=0, end_dim=-1):
    """Get size of batch dimension
    coord: [K... D]
    Returns
    -------
    torch.Tensor [(K...) D] : The trajectory with batch dimensions squeezed
    K...: The actual batch shapes
    """
    orig_shape = x.shape
    if slicelen(len(x.shape), start=start_dim, end=end_dim) > 0:
        x = torch.flatten(x, start_dim, end_dim)
    return x, orig_shape


def unflatten(x, orig_shape):
    return torch.reshape(x, orig_shape)


def _nufft(
    input: torch.Tensor,
    coord: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    upsampfac: float = 2.0,
) -> torch.Tensor:
    """
    input : torch.Tensor
        Shape [N... *im_size]
    coord : torch.Tensor
        Has scaling [-pi/2, pi/2]. Shape [K... D]
    upsampfac : float
        Upsampling factor for grid
    Returns
    -------
    output : torch.Tensor
        [N... K...]
    """
    dev = "cpu" if input.device == torch.device("cpu") else "gpu"
    if dev == "cpu":
        kwargs = {"spread_kerevalmeth": 1 if upsampfac == 2.0 else 0}
    else:
        kwargs = {"gpu_kerevalmeth": 1 if upsampfac == 2.0 else 0}
    dim = coord.shape[-1]
    nK = len(coord.shape[:-1])
    flat_input, input_shape = flatten(input, start_dim=0, end_dim=-(dim + 1))
    flat_coord, coord_shape = flatten(coord, start_dim=0, end_dim=-2)
    flat_out = None

    if out is not None:
        flat_out, out_shape = flatten(out, start_dim=0, end_dim=-(nK + 1))
        flat_out, _ = flatten(flat_out, start_dim=1, end_dim=-1)

    nufft_fn = get_nufft[dev][dim][0]

    coord_components = coord2contig(flat_coord)
    if dev == "cpu":
        coord_components = tuple(c.detach().numpy() for c in coord_components)
        flat_input = flat_input.detach().numpy()

    if flat_out is not None:
        nufft_fn(
            *coord_components,
            flat_input,
            isign=-1,
            out=flat_out,
            upsampfac=upsampfac,
            **kwargs,
        ) / sqrt(prod(input_shape[-dim:]))
    else:
        flat_out = nufft_fn(
            *coord_components,
            flat_input,
            isign=-1,
            upsampfac=upsampfac,
            **kwargs,
        ) / sqrt(prod(input_shape[-dim:]))

    if dev == "cpu":
        flat_out = torch.from_numpy(flat_out)
    output = unflatten(flat_out, (*input_shape[:-dim], *coord_shape[:-1]))
    return output


def _nufft_adjoint(
    input: torch.Tensor,
    coord: torch.Tensor,
    oshape: Tuple,
    out: Optional[torch.Tensor] = None,
    upsampfac: float = 2.0,
) -> torch.Tensor:
    """
    input : torch.Tensor
        Shape [N... K...]
    coord : torch.Tensor
        Shape [K... D], has scaling [-pi/2, pi/2]
    oshape : Tuple
        Desired output image shape (with batch dimensinos).
    out : Optional[torch.Tensor]
        Shape [N..., *im_size] optional output image
    upsampfac : float
        Upsampling factor for regular grid

    Returns
    -------
    output : torch.Tensor
        [N... *oshape]
    """
    dev = "cpu" if input.device == torch.device("cpu") else "gpu"
    if dev == "cpu":
        kwargs = {"spread_kerevalmeth": 1 if upsampfac == 2.0 else 0}
    else:
        kwargs = {"gpu_kerevalmeth": 1 if upsampfac == 2.0 else 0}
    dim = coord.shape[-1]
    coord_batch_len = len(coord.shape) - 1
    # out_batch = input.shape[:-coord_batch_len]
    flat_input, _ = flatten(input, start_dim=0, end_dim=-(coord_batch_len + 1))
    flat_input, _ = flatten(flat_input, start_dim=-coord_batch_len, end_dim=-1)
    flat_coord, _ = flatten(coord, start_dim=0, end_dim=-2)
    flat_out = None
    if out is not None:
        flat_out, out_shape = flatten(out, start_dim=0, end_dim=-(coord_batch_len + 1))

    adj_nufft_fn = get_nufft[dev][dim][1]

    coord_components = coord2contig(flat_coord)
    if dev == "cpu":
        coord_components = tuple(c.detach().numpy() for c in coord_components)
        flat_input = flat_input.detach().numpy()

    im_size = oshape[-dim:]
    if flat_out is not None:
        adj_nufft_fn(
            *coord_components,
            flat_input,
            im_size,
            isign=1,
            out=flat_out,
            upsampfac=upsampfac,
            **kwargs,
        ) / sqrt(prod(im_size))
    else:
        flat_out = adj_nufft_fn(
            *coord_components,
            flat_input,
            im_size,
            isign=1,
            upsampfac=upsampfac,
            **kwargs,
        ) / sqrt(prod(im_size))

    if dev == "cpu":
        flat_out = torch.from_numpy(flat_out)
    output = unflatten(flat_out, oshape)
    return output


class FiNUFFT(Function):
    @staticmethod
    def forward(
        input: torch.Tensor,
        coord: torch.Tensor,
        out: torch.Tensor,
        upsampfac: float,
    ) -> torch.Tensor:
        return _nufft(input, coord, out, upsampfac)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, coord, _, upsampfac = inputs
        ctx.oshape = input.shape
        ctx.upsampfac = upsampfac
        ctx.save_for_backward(coord)

    @staticmethod
    def backward(ctx, grad_output):
        coord = ctx.saved_tensors[0]
        grad_input = grad_coord = grad_out = grad_upsampfac = None

        if ctx.needs_input_grad[0]:
            grad_input = _nufft_adjoint(
                grad_output,
                coord,
                ctx.oshape,
                upsampfac=ctx.upsampfac,
            )
        return grad_input, grad_coord, grad_out


def nufft(input, coord, out=None, upsampfac=2.0):
    """
    Wrap to provide default and keyword arguments.
    """
    return FiNUFFT.apply(input, coord, out, upsampfac)


class FiNUFFTAdjoint(Function):
    @staticmethod
    def forward(
        input: torch.Tensor,
        coord: torch.Tensor,
        oshape: torch.Size,
        out: Optional[torch.Tensor],
        upsampfac: float,
    ) -> torch.Tensor:
        return _nufft_adjoint(input, coord, oshape, out, upsampfac)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, coord, _, _, upsampfac = inputs
        ctx.upsampfac = upsampfac
        ctx.save_for_backward(coord)

    @staticmethod
    def backward(ctx, grad_output):
        coord = ctx.saved_tensors[0]
        grad_input = grad_coord = grad_oshape = grad_out = grad_upsampfac = None

        if ctx.needs_input_grad[0]:
            grad_input = _nufft(grad_output, coord, upsampfac=ctx.upsampfac)
        return grad_input, grad_coord, grad_oshape, grad_out, grad_upsampfac


def nufft_adjoint(input, coord, oshape, out=None, upsampfac=2.0):
    """
    Wrap to provide default and keyword arguments
    """
    return FiNUFFTAdjoint.apply(input, coord, oshape, out, upsampfac)
