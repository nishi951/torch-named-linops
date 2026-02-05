from jaxtyping import Integer
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import torchlinops.functional as F
from torchlinops.utils import default_to

from ..nameddim import ELLIPSES, NamedShape as NS, Shape
from .namedlinop import NamedLinop

__all__ = ["Sampling"]


class Sampling(NamedLinop):
    """Sample a tensor at some specified integer locations.

    ```
    Input: (batch_shape, input_shape)
    Output: (batch_shape, output_shape)
    ```

    """

    def __init__(
        self,
        idx: tuple[Integer[Tensor, "..."], ...],
        input_size: tuple[int, ...],
        output_shape: Optional[Shape] = None,
        input_shape: Optional[Shape] = None,
        batch_shape: Optional[Shape] = None,
    ):
        """
        Sampling: (B..., N...) -> (B..., M...)

        Parameters
        ----------
        idx : tuple[Integer[Tensor, "..."], ...]
            tuple of of D  tensors, each of shape [M...]
            One index for each "sampled" axis of the input tensor
            Use `canonicalize_idx` to turn a tensor of shape [M... D] to a D-tuple of index tensors.
            idx is in range [0, size-1]
        input_size : tuple[int, ...]
            Actual shape of the input interpolated tensor, without the batch dimensions.


        """
        dim = len(input_size)
        if len(idx) != dim:
            raise ValueError(
                f"Input size {input_size} doesn't match index with length {len(idx)}."
            )
        self.input_size = input_size
        batch_shape = default_to(("...",), batch_shape)
        input_shape = default_to(("...",), input_shape)
        output_shape = default_to(("...",), output_shape)
        shape = NS(batch_shape) + NS(input_shape, output_shape)
        super().__init__(shape)
        self._shape.batch_shape = batch_shape
        self._shape.input_shape = input_shape
        self._shape.output_shape = output_shape
        idx = F.ensure_tensor_indexing(idx, self.input_size)
        for d, (t, s) in enumerate(zip(idx, self.input_size)):
            if (t < 0).any() or (t >= s).any():
                raise ValueError(
                    f"Sampling index must lie within range [0, {s - 1}] but got range [{t.min().item()}, {t.max().item()}] for dim {d}"
                )
        self.idx = nn.ParameterList([nn.Parameter(i, requires_grad=False) for i in idx])

    @property
    def locs(self):
        """for compatibility with Interpolate linop"""
        return torch.stack(tuple(self.idx), dim=-1)

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

    @staticmethod
    def fn(sampling, x, /):
        return F.index(x, tuple(sampling.idx))

    @staticmethod
    def adj_fn(sampling, x, /):
        return F.index_adjoint(x, tuple(sampling.idx), sampling.input_size)

    def split_forward(self, ibatch, obatch):
        if self._shape.output_shape == ELLIPSES:
            # Cannot split if idx batch shape is not split
            return self
        return type(self)(
            self.split_idx(ibatch, obatch, self.idx),
            self.input_size,
            self._shape.output_shape,
            self._shape.input_shape,
            self._shape.batch_shape,
        )

    def split_idx(self, ibatch, obatch, idx):
        num_output_dims = len(idx[0].shape)
        if num_output_dims > 0:
            idx_slc = tuple(obatch[-num_output_dims:])
            return [i[idx_slc] for i in idx]
        return idx

    def register_shape(self, name, shape: tuple):
        self._shape[name] = shape

    def size(self, dim):
        if dim in self._shape.output_shape:
            dim_idx = self._shape.output_shape.index(dim)
            return self.locs.shape[dim_idx]
        elif dim in self._shape.input_shape:
            dim_idx = self._shape.input_shape.index(dim)
            return self.input_size[dim_idx]
        return None
