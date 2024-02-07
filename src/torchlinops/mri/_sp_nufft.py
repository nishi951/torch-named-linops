"""Sigpy backend for Autograd-Compatible NUFFT

Does not support differentiation w.r.t. coordinates
"""
import sigpy as sp
import torch
from torch.autograd import Function

__all__ = [
    'nufft', 'nufft_adjoint'
]

def complex_to_pytorch(arr, requires_grad: bool = True):
    arr = sp.to_pytorch(arr, requires_grad)
    return torch.view_as_complex(arr)

class SigpyNUFFT(Function):
    @staticmethod
    def forward(
            input: torch.Tensor,
            coord: torch.Tensor,
            oversamp,
            width,
    ) -> torch.Tensor:
        requires_grad = input.requires_grad
        input = sp.from_pytorch(input, iscomplex=False)
        coord = sp.from_pytorch(coord, iscomplex=False)
        output = sp.nufft(input, coord, oversamp, width)
        return complex_to_pytorch(output, requires_grad)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, coord, oversamp, width = inputs
        ctx.oshape = input.shape
        ctx.oversamp = oversamp
        ctx.width = width
        ctx.save_for_backward(coord)

    @staticmethod
    def backward(ctx, grad_output):
        coord = ctx.saved_tensors[0]
        grad_input = grad_coord = grad_oversamp = grad_width = None

        if ctx.needs_input_grad[0]:
            coord = sp.from_pytorch(coord, iscomplex=False)
            grad_output = sp.from_pytorch(grad_output, iscomplex=False)
            grad_input = sp.nufft_adjoint(
                grad_output, coord, ctx.oshape, ctx.oversamp, ctx.width
            )
            grad_input = complex_to_pytorch(grad_input)
        return grad_input, grad_coord, grad_oversamp, grad_width

def nufft(input, coord, oversamp: float = 1.25, width: int = 4):
    """
    Wrap to provide default and keyword arguments.
    """
    return SigpyNUFFT.apply(input, coord, oversamp, width)


class SigpyNUFFTAdjoint(Function):
    @staticmethod
    def forward(
            input: torch.Tensor,
            coord: torch.Tensor,
            oshape: torch.Size,
            oversamp: float,
            width: int,
    ) -> torch.Tensor:
        requires_grad = input.requires_grad
        input = sp.from_pytorch(input, iscomplex=False)
        coord = sp.from_pytorch(coord, iscomplex=False)
        output = sp.nufft_adjoint(input, coord, oshape, oversamp, width)
        return complex_to_pytorch(output, requires_grad)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, coord, _, oversamp, width = inputs
        ctx.oversamp = oversamp
        ctx.width = width
        ctx.save_for_backward(coord)

    @staticmethod
    def backward(ctx, grad_output):
        coord = ctx.saved_tensors[0]
        grad_input = grad_coord = grad_oshape = grad_oversamp = grad_width = None

        if ctx.needs_input_grad[0]:
            coord = sp.from_pytorch(coord, iscomplex=False)
            grad_output = sp.from_pytorch(grad_output, iscomplex=False)
            grad_input = sp.nufft(
                grad_output, coord, ctx.oversamp, ctx.width
            )
            grad_input = complex_to_pytorch(grad_input)
        return grad_input, grad_coord, grad_oshape, grad_oversamp, grad_width

def nufft_adjoint(input, coord, oshape, oversamp: float = 1.25, width: int = 4):
    """
    Wrap to provide default and keyword arguments
    """
    return SigpyNUFFTAdjoint.apply(input, coord, oshape, oversamp, width)
