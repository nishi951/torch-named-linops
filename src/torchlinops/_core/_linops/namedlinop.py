from copy import copy

import torch
import torch.nn as nn


from . import scalar
from . import chain

__all__ = [
    "NamedLinop",
]


class NamedLinop(nn.Module):
    """Base Class for all NamedLinops"""

    def __init__(self, ishape, oshape):
        """ishape and oshape are symbolic, not numeric
        They also change if the adjoint is taken (!)
        """
        super().__init__()
        self.ishape = ishape
        self.oshape = oshape

        self._adj = None
        self._normal = None

        self._suffix = ""

    # Change the call to self.fn according to the data
    def forward(self, x: torch.Tensor):
        return self.fn(x)

    # Override
    def fn(self, x: torch.Tensor, /, data=None):
        """Placeholder for functional forwa rd operator.
        Non-input arguments should be keyword-only
        self can still be used - kwargs should contain elements
        that may change frequently (e.g. trajectories) and can
        ignore hyperparameters (e.g. normalization modes)
        """
        return x

    # Override
    def adj_fn(self, x: torch.Tensor, /, data=None):
        """Placeholder for functional adjoint operator.
        Non-input arguments should be keyword-only"""
        return x

    # Override
    def normal_fn(self, x: torch.Tensor, /, data=None):
        """Placeholder for efficient functional normal operator"""
        return self.adj_fn(self.fn(x, data), data)

    # Override
    def split_forward(self, ibatch, obatch):
        """Return a new instance"""
        raise NotImplementedError(f"{type(self).__name__} cannot be split.")

    # Override
    def split_forward_fn(self, ibatch, obatch, /, data=None):
        """Return data"""
        raise NotImplementedError(f"{type(self).__name__} cannot be split.")

    # Override
    def size(self, dim: str):
        """Get the size of a particular dim, or return
        None if this linop doesn't determine the size
        """
        return None

    # Override
    def size_fn(self, dim: str, /, data=None):
        """Functional version of size. Determines sizes from kwargs
        kwargs should be the same as the inputs to fn or adj_fn
        Return None if this linop doesn't determine the size of dim
        """
        return None

    # Probably don't override these
    @property
    def dims(self):
        return set(self.ishape).union(set(self.oshape))

    @property
    def H(self):
        """Adjoint operator"""
        if self._adj is None:
            self._adj = [self.get_adjoint()]  # Prevent registration as a submodule
        return self._adj[0]

    def get_adjoint(self):
        adj = copy(self)
        # Swap functions
        adj.fn, adj.adj_fn = self.adj_fn, self.fn
        adj.split, adj.adj_split = self.adj_split, self.split
        adj.split_fn, adj.adj_split_fn = self.split_fn, self.adj_split_fn
        # Swap shapes
        adj.ishape, adj.oshape = self.oshape, self.ishape
        adj._suffix += ".H"
        return adj

    @property
    def N(self):
        """Normal operator
        Note that the naive normal operator can always be created
        via A.H @ A. Therefore, this function is reserved
        for custom behavior, as many functions have optimized normal
        forms.
        """
        if self._normal is None:
            #     _normal = copy(self)
            #     _normal._suffix += '.N'
            #     self.normal = _normal
            # return self._normal
            self._normal = [self.get_normal()]  # Prevent registration as a submodule
        return self._normal[0]

    def get_normal(self, inner=None):
        """
        inner: Optional linop for toeplitz embedding
        """
        normal = copy(self)
        normal.fn = self.normal_fn
        normal.adj_fn = self.normal_fn

        def new_normal(x, *args, **kwargs):
            x = self.normal_fn(x, *args, **kwargs)
            return self.normal_fn(x, *args, **kwargs)

        normal.normal_fn = new_normal
        normal.ishape = self.ishape
        normal.oshape, normal.ishape = self.ishape, self.ishape
        normal._suffix += ".N"
        return normal

    def split(self, ibatch, obatch):
        """Return a split version of the linop such that`forward`
        performs a split version of the linop
        ibatch: tuple of slices of same length as ishape
        obatch: tuple of slices of same length as oshape
        """
        return self.split_forward(ibatch, obatch)

    def adj_split(self, ibatch, obatch):
        """Split the adjoint version"""
        return self.split_forward(obatch, ibatch).H

    def split_fn(self, ibatch, obatch, /, **kwargs):
        """Return split versions of the data that can be passed
        into fn and adj_fn to produce split versions
        """
        return self.split_forward_fn(ibatch, obatch, **kwargs)

    def adj_split_fn(self, ibatch, obatch, /, **kwargs):
        return self.split_forward_fn(obatch, ibatch, **kwargs)

    def flatten(self):
        """Get a flattened list of constituent linops for composition"""
        return [self]

    def compose(self, inner):
        """Do self AFTER inner"""
        before = inner.flatten()
        after = self.flatten()
        return chain.Chain(*(after + before))

    def __add__(self, right):
        ...

    def __radd__(self, left):
        ...

    def __mul__(self, right):
        if isinstance(right, float) or isinstance(right, torch.Tensor):
            right = scalar.Scalar(weight=right, ioshape=self.ishape)
            return self.compose(right)
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, float) or isinstance(left, torch.Tensor):
            left = scalar.Scalar(weight=left, ioshape=self.oshape)
            return left.compose(self)
        return NotImplemented

    def __matmul__(self, right):
        return self.compose(right)

    def __rmatmul__(self, left):
        return left.compose(self)

    def __repr__(self):
        """Helps prevent recursion error caused by .H and .N"""
        return (
            f"{self.__class__.__name__ + self._suffix}({self.ishape} -> {self.oshape})"
        )
