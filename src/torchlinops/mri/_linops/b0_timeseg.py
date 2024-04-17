from math import pi
from copy import copy
from typing import Literal, Tuple, Optional

from einops import repeat
import torch
import torch.nn as nn

from torchlinops import NamedLinop, Identity, get2dor3d, ND, NS


class B0Timeseg(NamedLinop):
    def __init__(
        self,
        phase_map: torch.Tensor,
        im_size: Tuple,
        in_batch_shape: Optional[Tuple] = None,
        b0_dim: str = "B",
    ):
        """
        Parameters
        ----------
        phase_map : torch.Tensor
            [num_segments, *im_size] Phase map. exp(-2pi*i*t*B0)
        in_batch_shape : Tuple
            Any input batch dims
        b0_dim : str or NamedDimension
            Name of the new dimension
        """
        self.im_size = im_size
        self.D = len(self.im_size)
        # self.ts = ts
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = (b0_dim,) + self.in_batch_shape
        ishape = self.in_batch_shape + get2dor3d(self.im_size)
        oshape = self.out_batch_shape + get2dor3d(self.im_size)
        super().__init__(NS(ishape, oshape))
        self.b0_dim = ND.infer(b0_dim)
        # self.ts = self.get_segment_ts(self.nro, self.dt, self.nseg)
        # phase_map = torch.exp(-2j * pi * self.b0_map * self.ts) # TODO check the sign on this
        self.phase_map = nn.Parameter(phase_map, requires_grad=False)

    @classmethod
    def from_b0_map(
        cls,
        b0_map: torch.Tensor,
        ts: torch.Tensor,
        in_batch_shape: Tuple,
        b0_dim: str = "B",
    ):
        """
        b0_map : torch.Tensor
            Shape [Nx Ny [Nz]] Off-resonance map in units of Hz
        ts : torch.Tensor
            1D tensor of representative times, one per segment
        in_batch_shape, b0_dim: needed for regular constructor
        """
        im_size = b0_map.shape
        in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        out_batch_shape = (b0_dim,) + in_batch_shape
        oshape = out_batch_shape + get2dor3d(im_size)
        ts = repeat(ts, "T -> T" + " ()" * (len(oshape) - 1))  # Unsqueeze image dims
        ts = ts.to(b0_map.device)
        phase_map = torch.exp(-2j * pi * b0_map * ts)  # TODO check the sign on this
        return cls(phase_map, im_size, in_batch_shape, b0_dim)

    @staticmethod
    def get_segment_ts(
        nro: int, dt: float, nseg: int, mode: Literal["center", "first"] = "center"
    ):
        """
        Parameters
        ----------
        nro : int
            Total number of readout points in un-segmented readout
        dt : float
            Sampling time in seconds between the readout points
        nseg : int
            Number of segments
        mode : 'center' or 'first'
            Whether to center the ts on the middle or first point of each segment

        Returns
        -------
        1D float torch.Tensor of representative times for each segment.
        """
        tseg = dt * float(nro) / nseg
        t0 = tseg / 2 if mode == "center" else 0.0
        segment_ts = [t0 + tseg * i for i in range(nseg)]
        return torch.tensor(segment_ts)

    def forward(self, x):
        return self.fn(self, x, self.phase_map)

    @staticmethod
    def fn(linop, x, /, phase_map):
        return x[None, ...] * phase_map

    @staticmethod
    def adj_fn(linop, y, /, phase_map):
        return torch.sum(y * torch.conj(phase_map), dim=0)

    def split_forward(self, ibatch, obatch):
        """Split over coil dim only"""
        for islc, oslc in zip(ibatch[-self.D :], obatch[-self.D :]):
            if islc != oslc:
                raise IndexError(
                    "B0Timeseg currently only supports matched image input/output slicing."
                )
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.phase_map),
            self.im_size,
            self.in_batch_shape,
            self.b0_dim,
        )

    def split_forward_fn(self, ibatch, obatch, /, phase_map):
        # first dim of obatch is always the time seg dim
        return phase_map[obatch[0]]

    def size(self, dim: str):
        return self.size_fn(dim, self.phase_map)

    def size_fn(self, dim: str, phase_map):
        if dim == self.b0_dim:  # Segment dim
            return phase_map.shape[0]
        elif dim in self.oshape[-self.D :]:  # Spatial dim
            return phase_map[0].shape[self.oshape[-self.D :].index(dim)]
        return None

    def normal(self, inner=None):
        if inner is None:
            return Identity(self.ishape)
        return super().normal(inner)
