"""(cu)fiNUFFT backend for Autograd-Compatible NUFFT

Does not support differentiation w.r.t. coordinates
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
# Type-3 NUFFT = "adjoint" NUFFT (nonuniform to uniform)
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


def flatten(x, start_dim=0, end_dim=-1):
    """Get size of batch dimension
    coord: [K... D]
    Returns
    -------
    torch.Tensor [(K...) D] : The trajectory with batch dimensions squeezed
    K...: The actual batch shapes
    """
    orig_shape = x.shape
    x = torch.flatten(x, start_dim, end_dim)
    return x, orig_shape


def unflatten(x, orig_shape):
    return torch.reshape(x, orig_shape)


def _nufft(input: torch.Tensor, coord: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    input : torch.Tensor
        Shape [N... *im_size]
    coord : torch.Tensor
        Has scaling [-pi/2, pi/2]. Shape [K... D]
    Returns
    -------
    output : torch.Tensor
        [N... K...]
    """
    dev = "cpu" if input.device == torch.device("cpu") else "gpu"
    dim = coord.shape[-1]
    flat_coord, coord_shape = flatten(coord, start_dim=0, end_dim=-2)
    flat_input, input_shape = flatten(input, start_dim=0, end_dim=-(dim + 1))

    nufft_fn = get_nufft[dev][dim][0]

    coord_components = coord2contig(flat_coord)
    if dev == "cpu":
        coord_components = tuple(c.detach().numpy() for c in coord_components)
        flat_input = flat_input.detach().numpy()

    output = nufft_fn(*coord_components, flat_input, out=out) / sqrt(prod(input_shape[-dim:]))

    if dev == "cpu":
        output = torch.from_numpy(output)
    output = unflatten(output, (*input_shape[:-dim], *coord_shape[:-1]))
    return output


def _nufft_adjoint(
        input: torch.Tensor, coord: torch.Tensor, oshape: Tuple, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    input : torch.Tensor
        Shape [N... K...]
    coord : torch.Tensor
        Shape [K... D], has scaling [-pi/2, pi/2]
    oshape : Tuple
       Desired output shape.

    Returns
    -------
    output : torch.Tensor
        [N... *oshape]
    """
    dev = "cpu" if input.device == torch.device("cpu") else "gpu"
    dim = coord.shape[-1]
    coord_batch_len = len(coord.shape) - 1
    # out_batch = input.shape[:-coord_batch_len]
    flat_coord, coord_shape = flatten(coord, start_dim=0, end_dim=-2)
    flat_input, _ = flatten(input, start_dim=0, end_dim=-(coord_batch_len + 1))
    flat_input, _ = flatten(flat_input, start_dim=-coord_batch_len, end_dim=-1)

    adj_nufft_fn = get_nufft[dev][dim][1]

    coord_components = coord2contig(flat_coord)
    if dev == "cpu":
        coord_components = tuple(c.detach().numpy() for c in coord_components)
        flat_input = flat_input.detach().numpy()

    im_size = oshape[-dim:]
    output = adj_nufft_fn(*coord_components, flat_input, im_size, out=out) / sqrt(prod(im_size))

    if dev == "cpu":
        output = torch.from_numpy(output)
    output = unflatten(output, oshape)
    return output


class FiNUFFT(Function):
    @staticmethod
    def forward(
        input: torch.Tensor,
        coord: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        return _nufft(input, coord, out)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, coord, _ = inputs
        ctx.oshape = input.shape
        ctx.save_for_backward(coord)

    @staticmethod
    def backward(ctx, grad_output):
        coord = ctx.saved_tensors[0]
        grad_input = grad_coord = grad_out = None

        if ctx.needs_input_grad[0]:
            grad_input = _nufft_adjoint(
                grad_output,
                coord,
                ctx.oshape,
            )
        return grad_input, grad_coord, grad_out


def nufft(input, coord, out=None):
    """
    Wrap to provide default and keyword arguments.
    """
    return FiNUFFT.apply(input, coord, out)


class FiNUFFTAdjoint(Function):
    @staticmethod
    def forward(
        input: torch.Tensor,
        coord: torch.Tensor,
        oshape: torch.Size,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return _nufft_adjoint(input, coord, oshape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, coord, _, _= inputs
        ctx.save_for_backward(coord)

    @staticmethod
    def backward(ctx, grad_output):
        coord = ctx.saved_tensors[0]
        grad_input = grad_coord = grad_oshape = grad_out = None

        if ctx.needs_input_grad[0]:
            grad_input = _nufft(grad_output, coord)
        return grad_input, grad_coord, grad_oshape, grad_out


def nufft_adjoint(input, coord, oshape, out=None):
    """
    Wrap to provide default and keyword arguments
    """
    return FiNUFFTAdjoint.apply(input, coord, oshape, out)
