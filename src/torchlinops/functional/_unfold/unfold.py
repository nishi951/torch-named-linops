from typing import Optional
from jaxtyping import Shaped, Bool, Float
from torch import Tensor

from itertools import product

import triton
import triton.language as tl
import torch

from .nblocks import get_nblocks

__all__ = ["unfold"]

# Maximum size of the cuda grid per dimension
# 3 dimensions maximum
MAX_GRID_PER_DIM = 1024

MAX_TENSOR_POW_OF_2 = 20


def unfold(
    x: Shaped[Tensor, "..."],
    block_size: tuple,
    stride: Optional[tuple] = None,
    mask: Optional[Bool[Tensor, "..."]] = None,
) -> Tensor:
    """Wrapper that dispatches complex and real tensors
    Also precomputes some shapes

    Parameters
    ----------
    x : Tensor
        Shape [B..., *im_size]

    Returns
    -------
    Tensor: Shape [B..., *blocks, *block_size]
        If mask is not None, block_size will be an int equal to the number of True elements in the mask
        Otherwise it will be the full block shape.


    """
    x_flat, shapes = prep_unfold_shapes(x, block_size, stride, mask)
    if torch.is_complex(x_flat):
        x_flat = torch.view_as_real(x_flat)
        x_flat = torch.flatten(x_flat, -2, -1)  # Flatten real/imag into last dim
        y_flat = _unfold(x_flat, **shapes)
        y = y_flat.reshape(
            *shapes["batch_shape"],
            *shapes["nblocks"],
            *shapes["block_size"],
        )
        y = y.reshape(*y.shape[:-1], y.shape[-1] // 2, 2)
        y = torch.view_as_complex(y)
    else:
        y_flat = _unfold(x_flat, **shapes)
        y = y_flat.reshape(
            *shapes["batch_shape"],
            *shapes["nblocks"],
            *shapes["block_size"],
        )
    if mask is not None:
        y = y[..., mask]
    return y


def _unfold(
    x: Shaped[Tensor, "B ..."],
    block_size: tuple[int, ...],
    stride: tuple[int, ...],
    ndim: int,
    im_size: tuple[int, ...],
    nblocks: tuple[int, ...],
    nbatch: int,
    **kwargs,
) -> Shaped[Tensor, "B ..."]:
    """Implementation of unfold"""
    if x.is_cuda and ndim in UNFOLD.keys():
        x = x.contiguous()  # Ensure contiguity
        with torch.cuda.device(x.device):
            # Allocate output
            y = torch.zeros(
                nbatch,
                *nblocks,
                *block_size,
                device=x.device,
                dtype=x.dtype,
            )
            grid = _get_grid(ndim, nbatch, nblocks)
            BLOCK_SIZE = tuple(
                min(
                    triton.next_power_of_2(blk_size),
                    2 ** (MAX_TENSOR_POW_OF_2 // ndim),
                )
                for blk_size in block_size
            )
            UNFOLD[ndim][grid](
                x,
                y,
                nbatch,
                *nblocks,
                *block_size,
                *im_size,
                *stride,
                *BLOCK_SIZE,
            )
    else:
        y = _unfold_torch(x, block_size, stride, ndim, im_size, nblocks, nbatch)
    return y


def _get_grid(ndim: int, nbatch, nblocks: tuple[int, ...]):
    if ndim == 1:
        grid = lambda meta: (  # noqa: E731
            nbatch * triton.cdiv(nblocks[0], meta["x_blocks_per_grid"]),
        )
    elif ndim == 2:
        grid = lambda meta: (  # noqa: E731
            nbatch * triton.cdiv(nblocks[0], meta["x_blocks_per_grid"]),
            triton.cdiv(nblocks[1], meta["y_blocks_per_grid"]),
        )
    elif ndim == 3:
        grid = lambda meta: (  # noqa: E731
            nbatch * triton.cdiv(nblocks[0], meta["x_blocks_per_grid"]),
            triton.cdiv(nblocks[1], meta["y_blocks_per_grid"]),
            triton.cdiv(nblocks[2], meta["z_blocks_per_grid"]),
        )
    else:
        raise ValueError(f"Invalid ndim = {ndim}")
    return grid


@triton.heuristics(
    values={
        "x_blocks_per_grid": lambda args: max(
            1, triton.cdiv(args["x_nblocks"], MAX_GRID_PER_DIM)
        ),
        "x_BLOCKS_per_block": lambda args: max(
            1, triton.cdiv(args["x_block_dim"], 2**MAX_TENSOR_POW_OF_2)
        ),
    },
)
@triton.jit
def _unfold1d(
    in_ptr,
    out_ptr,
    # Number of batches
    nbatch: int,
    # Number of blocks
    x_nblocks: int,
    # Size of each block
    x_block_dim: int,
    # Size of the input data
    x_size: int,
    # Stride of the blocks
    x_stride: int,
    # Size of the triton block (power of 2)
    X_BLOCK_SIZE: tl.constexpr,
    # Number of blocks per grid pid
    x_blocks_per_grid: int,
    # Number of triton blocks per block
    x_BLOCKS_per_block: int,
):
    """
    Note: Cannot use make_block_ptr for out_ptr because the output block
    might require masking.
    """
    pid_0 = tl.program_id(0)
    # Batch index, Block index
    NBx = pid_0 * x_blocks_per_grid
    N, Bx = NBx // x_nblocks, NBx % x_nblocks

    in_size = x_size
    nblocks = x_nblocks
    block_dim = x_block_dim

    in_blk_ptr = tl.make_block_ptr(
        in_ptr,
        shape=(nbatch, x_size),
        strides=(in_size, 1),
        offsets=(N, Bx * x_stride),
        block_shape=(1, X_BLOCK_SIZE),
        order=(0, 1),
    )
    x_base_range = tl.arange(0, X_BLOCK_SIZE)
    for i in range(x_blocks_per_grid):
        if Bx + i < x_nblocks:
            for u in range(x_BLOCKS_per_block):
                x_range = x_base_range + u * X_BLOCK_SIZE
                x_mask = x_range < x_block_dim
                blk_range = x_range
                blk_mask = x_mask

                out_range = blk_range[None]
                out_mask = blk_mask[None]
                blk = load_subblock1d(in_blk_ptr, u, X_BLOCK_SIZE)
                # Save block to output
                out_offset = N * nblocks * block_dim + (Bx + i) * block_dim
                tl.store(out_ptr + out_offset + out_range, blk, out_mask)
        in_blk_ptr = tl.advance(in_blk_ptr, (0, x_stride))


@triton.jit
def load_subblock1d(in_blk_ptr, x_idx: int, X_BLOCK_SIZE: tl.constexpr):
    return tl.load(
        in_blk_ptr.advance((0, x_idx * X_BLOCK_SIZE)),
        boundary_check=(1,),
        padding_option="zero",
    )


@triton.heuristics(
    values={
        "x_blocks_per_grid": lambda args: max(
            1, triton.cdiv(args["x_nblocks"], MAX_GRID_PER_DIM)
        ),
        "y_blocks_per_grid": lambda args: max(
            1, triton.cdiv(args["y_nblocks"], MAX_GRID_PER_DIM)
        ),
        "x_BLOCKS_per_block": lambda args: max(
            1, triton.cdiv(args["x_block_dim"], 2 ** (MAX_TENSOR_POW_OF_2 // 2))
        ),
        "y_BLOCKS_per_block": lambda args: max(
            1, triton.cdiv(args["y_block_dim"], 2 ** (MAX_TENSOR_POW_OF_2 // 2))
        ),
    },
)
@triton.jit
def _unfold2d(
    in_ptr,
    out_ptr,
    # Number of batches
    nbatch: int,
    # Number of blocks
    x_nblocks: int,
    y_nblocks: int,
    # Size of each block
    x_block_dim: int,
    y_block_dim: int,
    # Size of the input data
    x_size: int,
    y_size: int,
    # Stride of the blocks
    x_stride: int,
    y_stride: int,
    # Size of the triton block (power of 2)
    X_BLOCK_SIZE: tl.constexpr,
    Y_BLOCK_SIZE: tl.constexpr,
    # Number of blocks per grid pid
    x_blocks_per_grid: int,
    y_blocks_per_grid: int,
    # Number of triton blocks per block
    x_BLOCKS_per_block: int,
    y_BLOCKS_per_block: int,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    # Batch index, Block index
    NBx = pid_0 * x_blocks_per_grid
    N, Bx = NBx // x_nblocks, NBx % x_nblocks
    By = pid_1 * y_blocks_per_grid

    # global sizes
    in_size = x_size * y_size
    nblocks = x_nblocks * y_nblocks
    block_dim = x_block_dim * y_block_dim

    in_blk_ptr = tl.make_block_ptr(
        in_ptr,
        shape=(nbatch, x_size, y_size),
        strides=(in_size, y_size, 1),
        offsets=(N, Bx * x_stride, By * y_stride),
        block_shape=(1, X_BLOCK_SIZE, Y_BLOCK_SIZE),
        order=(0, 1, 2),
    )
    x_base_range = tl.arange(0, X_BLOCK_SIZE)
    y_base_range = tl.arange(0, Y_BLOCK_SIZE)

    for i in range(x_blocks_per_grid):
        if Bx + i < x_nblocks:
            x_blk_offset = (Bx + i) * y_nblocks * block_dim
            in_blk_ptr_x = in_blk_ptr
            for j in range(y_blocks_per_grid):
                if By + j < y_nblocks:
                    y_blk_offset = (By + j) * block_dim
                    # Loop over blocks within block
                    for u in range(x_BLOCKS_per_block):
                        for v in range(y_BLOCKS_per_block):
                            x_range = x_base_range + u * X_BLOCK_SIZE
                            y_range = y_base_range + v * Y_BLOCK_SIZE
                            x_mask = x_range < x_block_dim
                            y_mask = y_range < y_block_dim
                            blk_range = (
                                x_range[:, None] * y_block_dim + y_range[None, :]
                            )
                            blk_mask = x_mask[:, None] & y_mask[None, :]
                            # out_offset = N * x_nblocks * x_block_dim + Bx * x_block_dim
                            # add batch dim
                            out_range = blk_range[None, :, :]
                            out_mask = blk_mask[None, :, :]
                            # blk = tl.load(in_blk_ptr)
                            blk = load_subblock2d(
                                in_blk_ptr, u, v, X_BLOCK_SIZE, Y_BLOCK_SIZE
                            )
                            out_offset = (
                                N * nblocks * block_dim + x_blk_offset + y_blk_offset
                            )
                            tl.store(out_ptr + out_offset + out_range, blk, out_mask)
                in_blk_ptr = tl.advance(in_blk_ptr, (0, 0, y_stride))
            in_blk_ptr = in_blk_ptr_x
        in_blk_ptr = tl.advance(in_blk_ptr, (0, x_stride, 0))


@triton.jit
def load_subblock2d(in_blk_ptr, x_idx, y_idx, X_BLOCK_SIZE, Y_BLOCK_SIZE):
    return tl.load(
        in_blk_ptr.advance((0, x_idx * X_BLOCK_SIZE, y_idx * Y_BLOCK_SIZE)),
        boundary_check=(1, 2),
        padding_option="zero",
    )


@triton.heuristics(
    values={
        "x_blocks_per_grid": lambda args: max(
            1, triton.cdiv(args["x_nblocks"], MAX_GRID_PER_DIM)
        ),
        "y_blocks_per_grid": lambda args: max(
            1, triton.cdiv(args["y_nblocks"], MAX_GRID_PER_DIM)
        ),
        "z_blocks_per_grid": lambda args: max(
            1, triton.cdiv(args["z_nblocks"], MAX_GRID_PER_DIM)
        ),
        "x_BLOCKS_per_block": lambda args: max(
            1, triton.cdiv(args["x_block_dim"], 2 ** (MAX_TENSOR_POW_OF_2 // 3))
        ),
        "y_BLOCKS_per_block": lambda args: max(
            1, triton.cdiv(args["y_block_dim"], 2 ** (MAX_TENSOR_POW_OF_2 // 3))
        ),
        "z_BLOCKS_per_block": lambda args: max(
            1, triton.cdiv(args["z_block_dim"], 2 ** (MAX_TENSOR_POW_OF_2 // 3))
        ),
    },
)
@triton.jit
def _unfold3d(
    in_ptr,
    out_ptr,
    # Number of batches
    nbatch: int,
    # Number of blocks
    x_nblocks: int,
    y_nblocks: int,
    z_nblocks: int,
    # Size of each block
    x_block_dim: int,
    y_block_dim: int,
    z_block_dim: int,
    # Size of the input data
    x_size: int,
    y_size: int,
    z_size: int,
    # Stride of the blocks
    x_stride: int,
    y_stride: int,
    z_stride: int,
    # Size of the triton block (power of 2)
    X_BLOCK_SIZE: tl.constexpr,
    Y_BLOCK_SIZE: tl.constexpr,
    Z_BLOCK_SIZE: tl.constexpr,
    # Number of blocks per grid pid
    x_blocks_per_grid: int,
    y_blocks_per_grid: int,
    z_blocks_per_grid: int,
    # Number of triton blocks per block
    x_BLOCKS_per_block: int,
    y_BLOCKS_per_block: int,
    z_BLOCKS_per_block: int,
):
    """"""
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)
    # Batch index, Block index
    NBx = pid_0 * x_blocks_per_grid
    N, Bx = NBx // x_nblocks, NBx % x_nblocks
    By = pid_1 * y_blocks_per_grid
    Bz = pid_2 * z_blocks_per_grid

    # global sizes
    in_size = x_size * y_size * z_size
    nblocks = x_nblocks * y_nblocks * z_nblocks
    block_dim = x_block_dim * y_block_dim * z_block_dim

    in_blk_ptr = tl.make_block_ptr(
        in_ptr,
        shape=(nbatch, x_size, y_size, z_size),
        strides=(in_size, y_size * z_size, z_size, 1),
        offsets=(N, Bx * x_stride, By * y_stride, Bz * z_stride),
        block_shape=(1, X_BLOCK_SIZE, Y_BLOCK_SIZE, Z_BLOCK_SIZE),
        order=(0, 1, 2, 3),
    )
    x_base_range = tl.arange(0, X_BLOCK_SIZE)
    y_base_range = tl.arange(0, Y_BLOCK_SIZE)
    z_base_range = tl.arange(0, Z_BLOCK_SIZE)

    for i in range(x_blocks_per_grid):
        if Bx + i < x_nblocks:
            x_blk_offset = (Bx + i) * y_nblocks * z_nblocks * block_dim
            in_blk_ptr_x = in_blk_ptr
            for j in range(y_blocks_per_grid):
                if By + j < y_nblocks:
                    y_blk_offset = (By + j) * z_nblocks * block_dim
                    in_blk_ptr_y = in_blk_ptr
                    for k in range(z_blocks_per_grid):
                        if Bz + k < z_nblocks:
                            z_blk_offset = (Bz + k) * block_dim
                            # Loop over subblocks
                            for u in range(x_BLOCKS_per_block):
                                for v in range(y_BLOCKS_per_block):
                                    for w in range(z_BLOCKS_per_block):
                                        x_range = x_base_range + u * X_BLOCK_SIZE
                                        y_range = y_base_range + v * Y_BLOCK_SIZE
                                        z_range = z_base_range + w * Z_BLOCK_SIZE
                                        x_mask = x_range < x_block_dim
                                        y_mask = y_range < y_block_dim
                                        z_mask = z_range < z_block_dim

                                        blk_range = (
                                            x_range[:, None, None] * y_block_dim
                                            + y_range[None, :, None]
                                        ) * z_block_dim + z_range[None, None, :]
                                        blk_mask = x_mask[:, None, None] & (
                                            y_mask[None, :, None]
                                            & z_mask[None, None, :]
                                        )

                                        out_range = blk_range[None]
                                        out_mask = blk_mask[None]
                                        # Load/Store
                                        blk = load_subblock3d(
                                            in_blk_ptr,
                                            u,
                                            v,
                                            w,
                                            X_BLOCK_SIZE,
                                            Y_BLOCK_SIZE,
                                            Z_BLOCK_SIZE,
                                        )
                                        out_offset = (
                                            N * nblocks * block_dim
                                            + x_blk_offset
                                            + y_blk_offset
                                            + z_blk_offset
                                        )
                                        tl.store(
                                            out_ptr + out_offset + out_range,
                                            blk,
                                            out_mask,
                                        )
                        in_blk_ptr = tl.advance(in_blk_ptr, (0, 0, 0, z_stride))
                    in_blk_ptr = in_blk_ptr_y
                in_blk_ptr = tl.advance(in_blk_ptr, (0, 0, y_stride, 0))
            in_blk_ptr = in_blk_ptr_x
        in_blk_ptr = tl.advance(in_blk_ptr, (0, x_stride, 0, 0))


@triton.jit
def load_subblock3d(
    in_blk_ptr, x_idx, y_idx, z_idx, X_BLOCK_SIZE, Y_BLOCK_SIZE, Z_BLOCK_SIZE
):
    return tl.load(
        in_blk_ptr.advance(
            (0, x_idx * X_BLOCK_SIZE, y_idx * Y_BLOCK_SIZE, z_idx * Z_BLOCK_SIZE)
        ),
        boundary_check=(1, 2, 3),
        padding_option="zero",
    )


UNFOLD = {1: _unfold1d, 2: _unfold2d, 3: _unfold3d}


def _unfold_torch(
    x: Float[Tensor, "B ..."],
    block_size: tuple[int, ...],
    stride: tuple[int, ...],
    ndim: int,
    im_size: tuple[int, ...],
    nblocks: tuple[int, ...],
    nbatch: int,
) -> Float[Tensor, "B I ..."]:
    """Fallback option

    Note: Compile takes forever
    """
    out = torch.zeros((nbatch, *nblocks, *block_size), device=x.device, dtype=x.dtype)
    # Python implementation
    for batch in range(nbatch):
        for blk in product(*(range(nblk) for nblk in nblocks)):
            blk_slc = tuple(
                slice(iblk * st, iblk * st + blk_sz)
                for iblk, st, blk_sz in zip(blk, stride, block_size)
            )
            out_idx = (batch, *blk)
            in_idx = (batch, *blk_slc)
            out[out_idx] = x[in_idx]
    return out


def prep_unfold_shapes(
    x,
    block_size: tuple,
    stride: Optional[tuple] = None,
    mask: Optional[Bool[Tensor, "..."]] = None,
):
    # Infer some shapes
    ndim = len(block_size)
    im_size = x.shape[-ndim:]
    stride = stride if stride is not None else (1,) * ndim
    nblocks = get_nblocks(im_size, block_size, stride)

    if torch.is_complex(x):
        im_size = list(im_size)
        im_size[-1] *= 2
        block_size = list(block_size)
        block_size[-1] *= 2
        stride = list(stride)
        stride[-1] *= 2

    # Add or infer batch dim
    batch_shape = x.shape[:-ndim]
    if ndim < len(x.shape):
        x_flat = x.flatten(0, len(x.shape) - ndim - 1)
    else:
        x_flat = x[None]
    nbatch = x_flat.shape[0]

    # Handle mask
    if mask is not None:
        mask = mask.to(x.device)

    return x_flat, {
        "ndim": ndim,
        "im_size": im_size,
        "stride": stride,
        "nblocks": nblocks,
        "nbatch": nbatch,
        "batch_shape": batch_shape,
        "mask": mask,
        "block_size": block_size,
    }
