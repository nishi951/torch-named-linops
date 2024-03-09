"""(cu)fiNUFFT backend for Autograd-Compatible NUFFT

Does not support differentiation w.r.t. coordinates
"""
from math import prod, sqrt
from typing import Tuple

import finufft as F
import cufinufft as cF

from einops import rearrange
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


def _nufft(input: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
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

    output = nufft_fn(*coord_components, flat_input) / sqrt(prod(input_shape[-dim:]))

    if dev == "cpu":
        output = torch.from_numpy(output)
    output = unflatten(output, (*input_shape[:-dim], *coord_shape[:-1]))
    return output

    # if dev == 'cpu':
    #     if dim == 2:
    #         F.nufft2d2(input, coord_squeezed, n_trans=n_trans)
    #     if dim == 3:
    #         F.nufft3d2()
    # if dev == 'gpu':
    #     if dim == 2:
    #         cF.nufft2d2(...)
    #     if dim == 3:
    #         cF.nufft3d2(...)


def _nufft_adjoint(
    input: torch.Tensor, coord: torch.Tensor, oshape: Tuple
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
    output = adj_nufft_fn(*coord_components, flat_input, im_size) / sqrt(prod(im_size))

    if dev == "cpu":
        output = torch.from_numpy(output)
    output = unflatten(output, oshape)
    return output


class FiNUFFT(Function):
    @staticmethod
    def forward(
        input: torch.Tensor,
        coord: torch.Tensor,
    ) -> torch.Tensor:
        return _nufft(input, coord)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, coord = inputs
        ctx.oshape = input.shape
        ctx.save_for_backward(coord)

    @staticmethod
    def backward(ctx, grad_output):
        coord = ctx.saved_tensors[0]
        grad_input = grad_coord = None

        if ctx.needs_input_grad[0]:
            grad_input = _nufft_adjoint(
                grad_output,
                coord,
                ctx.oshape,
            )
        return grad_input, grad_coord


def nufft(input, coord):
    """
    Wrap to provide default and keyword arguments.
    """
    return FiNUFFT.apply(input, coord)


class FiNUFFTAdjoint(Function):
    @staticmethod
    def forward(
        input: torch.Tensor,
        coord: torch.Tensor,
        oshape: torch.Size,
    ) -> torch.Tensor:
        return _nufft_adjoint(input, coord, oshape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, coord, _ = inputs
        ctx.save_for_backward(coord)

    @staticmethod
    def backward(ctx, grad_output):
        coord = ctx.saved_tensors[0]
        grad_input = grad_coord = grad_oshape = None

        if ctx.needs_input_grad[0]:
            grad_input = _nufft(grad_output, coord)
        return grad_input, grad_coord, grad_oshape


def nufft_adjoint(input, coord, oshape):
    """
    Wrap to provide default and keyword arguments
    """
    return FiNUFFTAdjoint.apply(input, coord, oshape)
