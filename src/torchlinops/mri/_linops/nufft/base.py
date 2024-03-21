from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchlinops._core._linops import NamedLinop
from torchlinops._core._shapes import get2dor3d


class NUFFTBase(NamedLinop):
    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        shared_batch_shape: Optional[Tuple] = None,
    ):
        """
        img (input) [S... N... Nx Ny [Nz]]
        trj: [S... K..., D] in sigpy style [-N/2, N/2]
        in_batch_shape : Tuple
            The shape of [N...] in img
        out_batch_shape : Tuple
            The shape of [K...] in trj.
        shared_batch_shape : Tuple
            The shape of [S...] in trj

        """
        self.setup_shapes(in_batch_shape, out_batch_shape, shared_batch_shape, im_size)
        self.trj = nn.Parameter(trj, requires_grad=False)
        self.im_size = im_size

        # Precompute
        self.D = len(im_size)

    def setup_shapes(in_batch_shape, out_batch_shape, shared_batch_shape, im_size):
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = (
            out_batch_shape if out_batch_shape is not None else tuple()
        )
        self.shared_batch_shape = (
            shared_batch_shape if shared_batch_shape is not None else tuple()
        )
        self.shared_dims = len(self.shared_batch_shape)
        ishape = self.shared_batch_shape + self.in_batch_shape + get2dor3d(im_size)
        oshape = self.shared_batch_shape + self.in_batch_shape + self.out_batch_shape
        super().__init__(ishape, oshape)

    def split_forward(self, ibatch, obatch):
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.trj),
            im_size=self.im_size,
            in_batch_shape=self.in_batch_shape,
            out_batch_shape=self.out_batch_shape,
            shared_batch_shape=self.shared_batch_shape,
        )

    def split_forward_fn(self, ibatch, obatch, /, trj):
        shared_batch = obatch[: self.shared_dims]
        kbatch = obatch[self.shared_dims + len(self.in_batch_shape) :]
        trj_slc = tuple(shared_batch + kbatch + [slice(None)])
        # trj_slc = obatch[:-1] + [slice(None)] + obatch[-1:]
        return trj[trj_slc]

    def size(self, dim: str):
        return self.size_fn(dim, self.trj)

    def size_fn(self, dim: str, trj):
        if dim in self.shared_batch_shape:
            idx = self.shared_batch_shape.index(dim)
        elif dim in self.out_batch_shape:
            idx = len(self.shared_batch_shape) + self.out_batch_shape.index(dim)
        else:
            return None
        return trj.shape[idx]

    def timeseg(self, num_segments, segment_dim):
        """
        Convert a NUFFT-style linop to a segmented linop

        Parameters
        ----------
        num_segments : int
            The number of time segments to split the trajectory into
        F : NamedLinop
            NUFFT or GriddedNUFFT linop with a .trj parameter of shape
            [..., readout_dim, spatial_dim]
        D : NamedLinop
            Optional DCF linop with .weight parameter of shape [..., readout_dim]

        Returns
        -------
        NamedLinop
            New linop that expects an additional shared batch shape with dimension segment_dim in position 0
            Still outputs ksp with the same output shape as before


        """

        # Split trajectory into segments
        def segment_helper(t, num_segments, dim):
            segments = t.chunk(num_segments, dim=dim)
            first_segment = segments[0]
            last_segment = segments[-1]
            last_segment_size = last_segment.shape[dim]
            num_to_truncate = first_segment.shape[dim] - last_segment.shape[dim]
            # Pad last segment
            last_segment = end_pad_with_zeros(
                last_segment, dim, first_segment.shape[dim] - last_segment.shape[dim]
            )
            segments[-1] = last_segment
            return torch.stack(*segments, dim=0), num_to_truncate

        trj, num_to_truncate = segment_helper(self.trj, num_segments, dim=-2)

        # Change expected input and output shapes
        segment_dim = ND.infer(segment_dim)
        shared_batch_shape = (segment_dim,) + self.shared_batch_shape
        out_batch_shape = list(self.out_batch_shape)
        out_batch_shape[-1] = out_batch_shape[-1].next_unused(out_batch_shape)
        out_batch_shape = tuple(out_batch_shape)

        F = copy(self)
        F.setup_shapes(self.in_batch_shape,
                       out_batch_shape,
                       shared_batch_shape,
                       self.im_size)
        F.trj = trj

        R_oshape = F.shared_batch_shape + F.in_batch_shape + F.out_batch_shape[-1:] + F.out_batch_shape[-1].next_unused(F.out_batch_shape)
        ostr = F.shared_batch_shape + F.in_batch_shape + F.out_batch_shape[-1:] + f"({segment_dim} {readout_dim})"
        R = Rearrange(istr=str(F.oshape),
                      ostr=str(ostr),
                      ishape=F.oshape,
                      oshape=R_oshape)
        T = Truncate(dim=-1, length=num_to_truncate, ishape=R_oshape, oshape=self.oshape)
        return T @ R @ F
