from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchlinops._core._shapes import get2dor3d
from torchlinops._core._linops import NamedLinop, NS, Diagonal

__all__ = ["SpatialBasis", "TemporalBasis"]


class SpatialBasis(NamedLinop):
    """Spatial Basis portion of low-rank field decomposition

    basis : torch.Tensor
        [L, *im_size] where L is the number of spatial bases

    """

    def __init__(
        self,
        spatial_basis,
        basis_dim: Optional[str] = "L",
        in_batch_shape: Optional[Tuple] = None,
    ):
        self.im_size = spatial_basis.shape[1:]
        im_shape = get2dor3d(self.im_size)
        shape = NS(in_batch_shape) + NS(tuple(), (basis_dim,)) + NS(im_shape)
        self.basis_ax = -(len(self.im_size) + 1)
        self.D = len(self.im_size)
        super().__init__(shape)
        self._shape.add("basis_dim", basis_dim)
        self._shape.add("in_batch_shape", in_batch_shape)
        self.basis = nn.Parameter(spatial_basis, requires_grad=False)

    @property
    def basis_dim(self):
        return self._shape.basis_dim

    @property
    def in_batch_shape(self):
        return self._shape.in_batch_shape

    def forward(self, x):
        return self.fn(self, x, self.basis)

    @staticmethod
    def fn(linop, x, /, basis):
        return x.unsqueeze(linop.basis_ax) * basis

    @staticmethod
    def adj_fn(linop, x, /, basis):
        return torch.sum(x * torch.conj(basis), dim=linop.basis_ax)

    def split_forward(self, ibatch, obatch):
        """Split over batch dim only"""
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.basis),
            self.basis_dim,
            self.ishape[: -self.D],
        )

    def split_forward_fn(self, ibatch, obatch, /, basis):
        for islc, oslc in zip(ibatch[-self.D :], obatch[-self.D :]):
            if islc != oslc:
                raise IndexError(
                    "SpatialBasis currently only supports matched image input/output slicing."
                )
        return basis[obatch[self.basis_ax :]]

    def size(self, dim: str):
        return self.size_fn(dim, self.basis)

    def size_fn(self, dim: str, basis):
        forward_oshape = (self.basis_dim,) + self.oshape[-self.D :]
        basis_shape = forward_oshape[self.basis_ax :]
        if dim in basis_shape:
            return basis.shape[basis_shape.index(dim)]
        return None

    def normal(self, inner=None):
        if inner is None:
            abs_basis = torch.sum(torch.abs(self.basis) ** 2, dim=0)
            normal = Diagonal(abs_basis, self.oshape[-self.D :])
            return normal
        return super().normal(inner)


class TemporalBasis(NamedLinop):
    """Temporal Basis portion of spatiotemporal low-rank field decomposition

    temporal_basis : torch.Tensor
        [L, K...] tensor where K is the shape of the trajectory, not including the dim
        Example: Kspace trajectory of shape [R T K D] -> basis should have shape [L R T K]

    """

    def __init__(
        self,
        temporal_basis,
        out_batch_shape,
        basis_dim: Optional[str] = "L",
        in_batch_shape: Optional[Tuple] = None,
    ):
        self.K_shape = temporal_basis.shape[1:]
        shape = NS(in_batch_shape) + NS((basis_dim,), tuple()) + NS(out_batch_shape)
        self.basis_ax = -(len(out_batch_shape) + 1)
        self.K = len(out_batch_shape)
        super().__init__(shape)
        self._shape.add("basis_dim", basis_dim)
        self._shape.add("in_batch_shape", in_batch_shape)
        self._shape.add("out_batch_shape", out_batch_shape)
        self.basis = nn.Parameter(temporal_basis, requires_grad=False)

    @property
    def basis_dim(self):
        return self._shape.basis_dim

    @property
    def in_batch_shape(self):
        return self._shape.in_batch_shape

    @property
    def out_batch_shape(self):
        return self._shape.out_batch_shape

    def forward(self, x):
        return self.fn(self, x, self.basis)

    @staticmethod
    def fn(linop, x, /, basis):
        return torch.sum(x * basis, dim=linop.basis_ax)

    @staticmethod
    def adj_fn(linop, x, /, basis):
        return x.unsqueeze(linop.basis_ax) * torch.conj(basis)

    def split_forward(self, ibatch, obatch):
        """Split over batch dim only"""
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.basis),
            self.out_batch_shape,
            self.basis_dim,
            self.in_batch_shape,
        )

    def split_forward_fn(self, ibatch, obatch, /, basis):
        for islc, oslc in zip(ibatch[-self.K :], obatch[-self.K :]):
            if islc != oslc:
                raise IndexError(
                    "TemporalBasis currently only supports matched image input/output slicing."
                )
        return basis[obatch[self.basis_ax :]]

    def size(self, dim: str):
        return self.size_fn(dim, self.basis)

    def size_fn(self, dim: str, basis):
        forward_oshape = (self.basis_dim,) + self.oshape[-self.K :]
        basis_shape = forward_oshape[self.basis_ax :]
        if dim in basis_shape:
            return basis.shape[basis_shape.index(dim)]
        return None

    def normal(self, inner=None):
        if inner is None:
            abs_basis = torch.sum(torch.abs(self.basis) ** 2, dim=0)
            normal = Diagonal(abs_basis, self.oshape[-self.K :])
            return normal
        return super().normal(inner)
