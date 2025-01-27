from typing import Optional
from torch import Tensor

import torch.nn as nn

from .namedlinop import NamedLinop
from torchlinops._core._linops.nameddim import NDorStr, ELLIPSES, NS
from torchlinops.utils import default_to
import torchlinops.functional as F

OptionalShape = Optional[tuple[NDorStr]]

__all__ = ["Sampling"]


class Sampling(NamedLinop):
    """Sampling linop"""

    def __init__(
        self,
        idx: tuple,
        input_size: tuple,
        output_shape: tuple[NDorStr],
        input_shape: OptionalShape = None,
        batch_shape: OptionalShape = None,
    ):
        """
        Sampling: (B..., N...) -> (B..., M...)

        Parameters
        ----------
        idx : tuple of [M...] tensors
            One index for each "sampled" axis of the input tensor
            Use `canonicalize_idx` to turn a tensor of shape [M... D] to a D-tuple of index tensors.


        """
        self.input_size = input_size
        batch_shape = default_to(("...",), batch_shape)
        input_shape = default_to(("...",), input_shape)
        output_shape = default_to(("...",), output_shape)
        shape = NS(batch_shape) + NS(input_shape, output_shape)
        super().__init__(shape)
        self.register_shape("input_shape", input_shape)
        self.register_shape("output_shape", output_shape)
        self.register_shape("batch_shape", batch_shape)
        if len(output_shape) != len(idx):
            raise ValueError(
                f"Output shape {output_shape} doesn't correspond to idx with shape {len(idx)}"
            )
        self.idx = nn.ParameterList([nn.Parameter(i, requires_grad=False) for i in idx])

    @classmethod
    def from_mask(cls, mask, *args, **kwargs):
        """Alternative constructor for mask-based sampling"""
        idx = F.mask2idx(mask.bool())
        return cls(idx, *args, **kwargs)

    @classmethod
    def from_stacked_idx(cls, idx: Tensor, *args, **kwargs):
        """Alternative constructor for index in [M... D] form"""
        idx = F.canonicalize_idx(idx)
        return cls(idx, *args, **kwargs)

    def forward(self, x):
        return self.fn(self, x, self.idx)

    @staticmethod
    def fn(linop, x, idx):
        return F.index(x, idx)

    @staticmethod
    def adj_fn(linop, x, idx):
        batch_ndims = len(x.shape) - len(idx)
        if batch_ndims < 0:
            raise ValueError(
                f"Negative number of batch dimensions from input with shape {x.shape} and sampling index with shape {idx.shape}"
            )
        oshape = (*x.shape[:batch_ndims], *linop.input_size)
        return F.index_adjoint(x, idx, oshape=oshape)

    def split_forward(self, ibatch, obatch):
        if self._shape.output_shape == ELLIPSES:
            # Cannot split if idx batch shape is not split
            return self
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.idx),
            self.input_size,
            self._shape.output_shape,
            self._shape.input_shape,
            self._shape.batch_shape,
        )

    def split_forward_fn(self, ibatch, obatch, idx):
        nM = len(idx)
        if nM > 0:
            idx_slc = list(obatch[-nM:])
            return [i[idx_slc] for i in idx]
        return idx

    def register_shape(self, name, shape: tuple):
        self._shape.add(name, shape)
