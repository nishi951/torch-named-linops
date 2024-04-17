from copy import copy
from typing import Literal, Optional

import torch
from einops import rearrange
from torchlinops._core._linops import (
    Diagonal,
    ND,
    Truncate,
    Rearrange,
)
from torchlinops import NS
from torchlinops.utils import end_pad_with_zeros


def segment_helper(t, num_segments, dim):
    """Splits a tensor into segments along that dimension.
    Segment dimension becomes 0th dim of output tensor

    Returns
    -------
    Reshaped stack of segmented dimensions
    Number of readout points to truncate from the last segment

    """
    segments = t.chunk(num_segments, dim=dim)
    first_segment = segments[0]
    last_segment = segments[-1]
    num_to_truncate = first_segment.shape[dim] - last_segment.shape[dim]
    # Pad last segment
    last_segment = end_pad_with_zeros(
        last_segment, dim, first_segment.shape[dim] - last_segment.shape[dim]
    )
    segments = segments[:-1] + (last_segment,)
    return torch.stack(segments, dim=0), num_to_truncate


def Z_segmented(nseg, num_readout_extra, num_readout, ioshape):
    """Make a linop that zeros out the last few readout points
    of a trajectory.
    Example: with 5 time segments and original trajectory with readout dim 450

    num_readout = 450

    5 Segments -> 4 segments of length 400, 1 segment of length 50, padded to 100
    num_readout_extra = 5 * 100 = 500

    new ksp has shape [5 ... 100]

    This function creates a diagonal linop Z with shape
    [5 1... 100]
    where Z.weight[4, ..., 450:] == 0 and all others are 1
    - i.e. the last 50 points of the weight are zero'd out
    - All others are equal to 1
    - Broadcast dims inside.
    """
    zero_out = torch.ones(num_readout_extra)
    zero_out[num_readout:] = 0.0
    n_broadcast = len(ioshape) - 2
    zero_out = rearrange(zero_out, "(B K) -> B" + " ()" * n_broadcast + "K", B=nseg)
    Z = Diagonal(
        weight=zero_out,
        ioshape=ioshape,
        broadcast_dims=ioshape[1:-1],
    )
    return Z


def F_segmented(Nufft, num_segments, segment_dim, readout_dim: Optional[str] = None):
    """
    Parameters
    ----------
    Nufft : NamedLinop
        NUFFT or GriddedNUFFT linop with a .trj parameter of shape
        [..., readout_dim, spatial_dim]
    num_segments : int
        The number of time segments to split the trajectory into
    segment_dim : str or NamedDimension
        The new name of the segmented dimension.
    readout_dim : str or NamedDimension
        The new name of the readout dimension (leave none to keep the same name)
    """
    segmented_trj, num_to_truncate = segment_helper(Nufft.trj, num_segments, dim=-2)

    # Change expected input and output shapes
    out_batch_shape = list(Nufft.out_batch_shape)
    if readout_dim is not None:
        out_batch_shape[-1] = readout_dim

    # Add segment dim at position 0 of oshape
    # Change name of segmented readout dim
    F = type(Nufft)(
        trj=segmented_trj,
        im_size=Nufft.im_size,
        shared_batch_shape=(segment_dim,) + Nufft.shared_batch_shape,
        in_batch_shape=Nufft.in_batch_shape,
        out_batch_shape=Nufft.out_batch_shape,
        extras=Nufft.extras,
        toeplitz=Nufft.toeplitz,
        toeplitz_oversamp=Nufft.toeplitz_oversamp,
    )
    return F


def D_segmented(DCF, num_segments, segment_dim, readout_dim: Optional[str] = None):
    """
    Parameters
    ----------
    DCF : NamedLinop
        Diagonal linop with weight of shape [..., readout_dim]
    num_segments : int
        The number of time segments to split the trajectory into
    segment_dim : str or NamedDimension
        The new name of the segmented dimension.
    readout_dim : str or NamedDimension
        The new name of the readout dimension. If None, keeps the same name.

    Returns
    -------
    Diagonal linop with new dcf shape.
    """
    segmented_weight, num_to_truncate = segment_helper(DCF.weight, num_segments, dim=-1)

    oshape = list(DCF.oshape)
    if readout_dim is not None:
        oshape[-1] = oshape[-1].next_unused(oshape)
    oshape = [segment_dim] + oshape
    broadcast_dims = list(DCF.broadcast_dims)
    while len(oshape) > len(segmented_weight.shape):
        new_broadcast_dim = oshape[-len(segmented_weight.shape)]
        if new_broadcast_dim not in broadcast_dims:
            broadcast_dims.append(new_broadcast_dim)
        segmented_weight = segmented_weight.unsqueeze(
            1
        )  # unsqueeze right after new segment dim

    D = Diagonal(segmented_weight, ioshape=oshape, broadcast_dims=broadcast_dims)
    return D


def timeseg(
    Nufft, num_segments, segment_dim, mode: Literal["truncate", "zero"] = "zero"
):
    """
    Convert a NUFFT-style linop to a segmented linop

    Parameters
    ----------
    Nufft : NamedLinop
        NUFFT or GriddedNUFFT linop with a .trj parameter of shape
        [..., readout_dim, spatial_dim]
    num_segments : int
        The number of time segments to split the trajectory into
    segment_dim : str or NamedDimension
        The new name of the segmented dimension.
    mode : 'truncate' or 'zero'
        If "truncate" - Trim the extra readout points from the new output.
        If "zero" - leave the readout points but set them to 0.

    Returns
    -------
    NamedLinop
        New linop that expects an additional shared batch shape with dimension segment_dim in position 0
        Still outputs ksp with the same output shape as before


    """
    new_readout_dim = Nufft.oshape[-1].next_unused(Nufft.oshape)
    F = F_segmented(Nufft, num_segments, segment_dim, new_readout_dim)

    # Recombine segment dim and segmented readout dim
    opattern = (
        " ".join(str(d) for d in F.oshape[1:-1]) + f" ({F.oshape[0]} {F.oshape[-1]})"
    )

    num_to_truncate = F.trj.shape[0] * F.trj.shape[-1] - Nufft.trj.shape[-2]
    if num_to_truncate > 0:
        if mode == "truncate":
            # Add another readout dim name for the
            # rearranged but not truncated readout dimension
            R_oshape = list(F.oshape)[1:]
            R_oshape[-1] = F.oshape[-1].next_unused(F.oshape)

            R = Rearrange(
                ipattern=" ".join(str(d) for d in F.oshape),
                opattern=opattern,
                ishape=F.oshape,
                oshape=R_oshape,
                axes_lengths={str(segment_dim): num_segments},
            )

            # Truncate readout dim to original length
            T = Truncate(
                dim=-1, length=num_to_truncate, ishape=R.oshape, oshape=Nufft.oshape
            )
            # Rearrange, then truncate
            return T @ R @ F
        elif mode == "zero":
            num_seg_readout = F.trj.shape[0] * F.trj.shape[-2]
            num_readout = Nufft.trj.shape[-2]
            Z = Z_segmented(num_segments, num_seg_readout, num_readout, F.oshape)
            return Z @ F
        else:
            raise ValueError(f"Unrecognized truncation mode {mode}")

    if mode == "truncate":
        R = Rearrange(
            ipattern=" ".join(str(d) for d in F.oshape),
            opattern=opattern,
            ishape=F.oshape,
            oshape=Nufft.oshape,
            axes_lengths={str(segment_dim): num_segments},
        )
        return R @ F
    elif mode == "zero":
        return F
    else:
        raise ValueError(f"Unrecognized truncation mode {mode}")
