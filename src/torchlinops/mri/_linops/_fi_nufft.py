"""(cu)fiNUFFT backend for Autograd-Compatible NUFFT

Does not support differentiation w.r.t. coordinates
"""
from math import prod
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


def squeeze(coord):
    """Get size of batch dimension
    coord: [K... D]
    Returns
    -------
    torch.Tensor [(K...) D] : The trajectory with batch dimensions squeezed
    K...: The actual batch shapes
    """
    batch_shape = coord.shape[:-1]
    return rearrange(coord, "... D -> (...) D"), batch_shape


def unsqueeze_output(output_squeezed, batch_shape):
    return torch.reshape(output_squeezed, batch_shape)


def _nufft(input: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    """
    coord has scaling [-pi/2, pi/2]
    """
    dev = "cpu" if input.device == torch.device("cpu") else "gpu"
    dim = coord.shape[-1]
    coord_squeezed, batch_shape = squeeze(coord)
    n_trans = prod(batch_shape)
    nufft_fn = get_nufft[dev][dim][0]

    coord_components = coord2contig(coord_squeezed)
    if dev == 'cpu':
        coord_components = tuple(c.detach().numpy() for c in coord_components)
        input = input.detach().numpy()

    output_squeezed = nufft_fn(*coord_components, input) / prod(input.shape[-dim:])

    if dev == 'cpu':
        output_squeezed = torch.from_numpy(output_squeezed)
    output = unsqueeze_output(output_squeezed, batch_shape)
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


def _nufft_adjoint(input: torch.Tensor, coord: torch.Tensor, oshape: Tuple) -> torch.Tensor:
    dev = "cpu" if input.device.idx >= 0 else "gpu"
    dim = coord.shape[-1]
    input_squeezed, coord_squeezed, batch_shape = squeeze(coord)
    n_trans = prod(batch_shape)
    adj_nufft_fn = get_nufft[dev][dim][1]
    output_squeezed = adj_nufft_fn(input, coord_squeezed, n_trans)
    output = unsqueeze_output(output_squeezed)
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
