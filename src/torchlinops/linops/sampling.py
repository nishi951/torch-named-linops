from typing import Optional
from torch import Tensor

import torch
import torch.nn as nn

from .namedlinop import NamedLinop
from .nameddim import NDorStr, ELLIPSES, NS, Shape
from torchlinops.utils import default_to
import torchlinops.functional as F
from torchlinops.functional._index.index import ensure_tensor_indexing


__all__ = ["Sampling"]


class Sampling(NamedLinop):
    """Sampling linop"""

    def __init__(
        self,
        idx: tuple,
        input_size: tuple,
        output_shape: Shape,
        input_shape: Optional[Shape] = None,
        batch_shape: Optional[Shape] = None,
    ):
        """
        Sampling: (B..., N...) -> (B..., M...)

        Parameters
        ----------
        idx : tuple of [M...] tensors
            One index for each "sampled" axis of the input tensor
            Use `canonicalize_idx` to turn a tensor of shape [M... D] to a D-tuple of index tensors.
            idx is in range [0, size-1]


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
        # if len(input_shape) != len(idx):
        #     raise ValueError(
        #         f"Input shape {input_shape} doesn't correspond to idx with shape {len(idx)}"
        #     )
        idx = ensure_tensor_indexing(idx, self.input_size)
        for d, (t, s) in enumerate(zip(idx, self.input_size)):
            if (t < 0).any() or (t >= s).any():
                raise ValueError(
                    f"Sampling index must lie within range [0, {s - 1}] but got range [{t.min().item()}, {t.max().item()}] for dim {d}"
                )
        self.idx = nn.ParameterList([nn.Parameter(i, requires_grad=False) for i in idx])

    @classmethod
    def from_mask(cls, mask, *args, **kwargs):
        """Alternative constructor for mask-based sampling"""
        idx = F.mask2idx(mask.bool())
        return cls(idx, *args, **kwargs)

    @classmethod
    def from_stacked_idx(cls, idx: Tensor, *args, dim=-1, **kwargs):
        """Alternative constructor for index in [M... D] form"""
        idx = F.canonicalize_idx(idx, dim=-1)
        return cls(idx, *args, **kwargs)

    def forward(self, x):
        return self.fn(self, x, tuple(self.idx))

    @staticmethod
    def fn(linop, x, idx):
        return F.index(x, idx)

    @staticmethod
    def adj_fn(linop, x, idx):
        return F.index_adjoint(x, idx, linop.input_size)

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
        nM = len(idx[0].shape)
        if nM > 0:
            idx_slc = list(obatch[-nM:])
            return [i[idx_slc] for i in idx]
        return idx

    def register_shape(self, name, shape: tuple):
        self._shape.add(name, shape)
