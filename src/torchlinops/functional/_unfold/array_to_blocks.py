"""Differentiable Block/Unblock functions"""

from typing import Optional, Literal
from jaxtyping import Bool
from torch import Tensor
from torch.autograd import Function

import torch
from .unfold import unfold
from .fold import fold

__all__ = ["array_to_blocks", "blocks_to_array"]


class ArrayToBlocksFn(Function):
    """Equal to block/unfold"""

    @staticmethod
    def forward(
        input: Tensor,
        block_shape,  # tuple
        stride,  # tuple
        mask,  # Bool[Tensor]
    ) -> Tensor:
        """Remembered something about Function not liking type annotations?"""
        output = unfold(input, block_shape, stride, mask)
        output.requires_grad_(input.requires_grad)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Unpack input and output
        input, block_shape, stride, mask = inputs

        # Save for backward pass
        ctx.im_size = tuple(input.shape[-len(block_shape) :])
        ctx.block_shape = block_shape
        ctx.stride = stride
        ctx.save_for_backward(mask)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_block_shape = grad_stride = grad_mask = None

        if ctx.needs_input_grad[0]:
            mask = ctx.saved_tensors[0]
            grad_input = fold(
                grad_output,
                ctx.im_size,
                ctx.block_shape,
                ctx.stride,
                mask,
            )
        return grad_input, grad_block_shape, grad_stride, grad_mask


def array_to_blocks(
    input,
    block_shape: tuple[int, ...],
    stride: Optional[tuple[int, ...]] = None,
    mask: Optional[Bool[Tensor, "..."]] = None,
):
    """Wrapper for default arguments"""
    return ArrayToBlocksFn.apply(input, block_shape, stride, mask)


class BlocksToArrayFn(Function):
    """Equal to block_adjoint/fold"""

    @staticmethod
    def forward(
        input: Tensor,
        im_size,  #  tuple
        block_shape,  # tuple
        stride,  # tuple
        mask,  # Tensor
    ) -> Tensor:
        output = fold(input, im_size, block_shape, stride, mask)
        output.requires_grad_(input.requires_grad)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, im_size, block_shape, stride, mask = inputs

        # Save for backward pass
        ctx.block_shape = block_shape
        ctx.stride = stride
        ctx.save_for_backward(mask)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_im_size = grad_block_shape = grad_stride = grad_mask = None
        if ctx.needs_input_grad[0]:
            mask = ctx.saved_tensors[0]
            grad_input = unfold(grad_output, ctx.block_shape, ctx.stride, mask)
        return grad_input, grad_im_size, grad_block_shape, grad_stride, grad_mask


def blocks_to_array(
    input,
    im_size: tuple,
    block_shape: tuple,
    stride: Optional[tuple] = None,
    mask: Optional[Bool[Tensor, "..."]] = None,
):
    """Wrapper for default arguments"""
    return BlocksToArrayFn.apply(input, im_size, block_shape, stride, mask)


def get_norm_weights(
    im_size: tuple[int, ...],
    block_shape: tuple[int, ...],
    stride: Optional[tuple[int, ...]] = None,
    mask: Optional[Bool[Tensor, "..."]] = None,
    device: torch.device = "cpu",
):
    """Compute normalizing weights

    1./weights * blocks_to_array(array_to_blocks(x)) == x

    """
    x = torch.ones(im_size, device=device)
    Ax = array_to_blocks(x, block_shape, stride, mask)
    weights = blocks_to_array(Ax, im_size, block_shape, stride, mask)
    return weights
