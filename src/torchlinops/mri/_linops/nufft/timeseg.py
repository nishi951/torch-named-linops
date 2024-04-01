import torch

from torchlinops._core._linops import (
    ND,
    Truncate,
    Rearrange,
)
from torchlinops.utils import end_pad_with_zeros


def segment_helper(t, num_segments, dim):
    """Splits a tensor into segments along that dimension.
    Segment dimension becomes 0th dim of output tensor
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


def timeseg(Nufft, num_segments, segment_dim):
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

    Returns
    -------
    NamedLinop
        New linop that expects an additional shared batch shape with dimension segment_dim in position 0
        Still outputs ksp with the same output shape as before


    """
    segmented_trj, num_to_truncate = segment_helper(Nufft.trj, num_segments, dim=-2)
    # Change expected input and output shapes
    segment_dim = ND.infer(segment_dim)
    segment_readout_dim = ND.infer(Nufft.out_batch_shape)[-1].next_unused(
        Nufft.out_batch_shape
    )
    segmented_shared_batch_shape = (segment_dim,) + ND.infer(Nufft.shared_batch_shape)
    segmented_out_batch_shape = ND.infer(Nufft.out_batch_shape)[:-1] + (
        segment_readout_dim,
    )

    # Add segment dim at position 0 of oshape
    # Change name of segmented readout dim
    F = type(Nufft)(
        segmented_trj,
        Nufft.im_size,
        Nufft.in_batch_shape,
        segmented_out_batch_shape,
        segmented_shared_batch_shape,
        Nufft.extras,
        Nufft.toeplitz,
        Nufft.toeplitz_oversamp,
    )

    # Recombine segment dim and segmented readout dim
    opattern = (
        " ".join(str(d) for d in F.oshape[1:-1])
        + f" ({segment_dim} {segment_readout_dim})"
    )
    if num_to_truncate > 0:
        # Add another readout dim name for the
        # rearranged but not truncated readout dimension
        extended_readout_dim = segment_readout_dim.next_unused(F.out_batch_shape)
        R_oshape = F.oshape[1:-1] + (extended_readout_dim,)

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
        return T @ R @ F
    R = Rearrange(
        ipattern=" ".join(str(d) for d in F.oshape),
        opattern=opattern,
        ishape=F.oshape,
        oshape=Nufft.oshape,
        axes_lengths={str(segment_dim): num_segments},
    )
    return R @ F
