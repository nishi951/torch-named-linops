from copy import copy, deepcopy
import traceback
import logging

import torch
import torch.nn as nn

import torchlinops

from .nameddim import NamedDimension as ND, NamedShape, NS

__all__ = [
    "NamedLinop",
]

logger = logging.getLogger(__name__)


class NamedLinop(nn.Module):
    """Base Class for all NamedLinops"""

    def __init__(self, shape: NamedShape):
        super().__init__()
        self._shape = shape

        self.reset()

        self._suffix = ""

    # Change the call to self.fn according to the data
    def forward(self, x: torch.Tensor):
        return self.fn(self, x)

    # Override
    @staticmethod
    def fn(linop, x: torch.Tensor, /, data=None):
        """Functional forward operator.
        Non-input arguments should be keyword-only
        self can still be used - kwargs should contain elements
        that may change frequently (e.g. trajectories) and can
        ignore hyperparameters (e.g. normalization modes)

        Staticmethod because it needs to be unbound to swap for the adjoint

        """
        return x

    # Override
    @staticmethod
    def adj_fn(linop, x: torch.Tensor, /, data=None):
        """Placeholder for functional adjoint operator.
        Non-input arguments should be keyword-only

        Staticmethod because it needs to be unbound to swap for adjoint
        """
        return x

    # Override
    # @staticmethod
    def normal_fn(self, x: torch.Tensor, /, data=None):
        """Placeholder for efficient functional normal operator
        Staticmethod because it needs to be unbound to swap for normal
        """
        return self.adj_fn(self.fn(x, data), data)

    # Override
    def split_forward(self, ibatch, obatch):
        """Return a new instance"""
        return type(self)(self._shape)

    # Override
    def split_forward_fn(self, ibatch, obatch, /, data=None):
        """Return data"""
        return None

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
        """Adjoint operator, with caching"""
        if self._adjoint is None:
            try:
                _adjoint = self.adjoint()
                _adjoint._adjoint = [self]
                self._adjoint = [_adjoint]  # Prevent registration as a submodule
            except AttributeError as e:
                traceback.print_exc()
                raise e
            logger.debug(f"{type(self).__name__}: Making new adjoint {_adjoint._shape}")
        return self._adjoint[0]

    def adjoint(self):
        """Create a new adjoint linop"""
        adj = copy(self)  # Retains data
        adj._shape = adj._shape.H
        # Swap functions (requires staticmethod)
        adj.fn, adj.adj_fn = adj.adj_fn, adj.fn
        adj.split, adj.adj_split = adj.adj_split, adj.split
        adj.split_fn, adj.adj_split_fn = adj.split_fn, adj.adj_split_fn
        if adj._suffix.endswith(".H"):
            adj._suffix = adj._suffix[:-2]
        else:
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
            try:
                _normal = self.normal()
                self._normal = [_normal]
            except AttributeError as e:
                traceback.print_exc()
                raise e
        return self._normal[0]

    def normal(self, inner=None):
        """Create a new normal linop
        inner: Optional linop for toeplitz embedding
        TODO: Add splitting for normal ops created this way.
        """
        if inner is None:
            normal = copy(self)
            normal._shape = self._shape.N

            def new_forward_adjoint_fn(linop, x, *args, **kwargs):
                return self.normal_fn(self, x, *args, **kwargs)

            normal.fn = new_forward_adjoint_fn
            normal.adj_fn = new_forward_adjoint_fn

            def new_normal_fn(linop, x, *args, **kwargs):
                AHAx = self.normal_fn(self, x, *args, **kwargs)
                return self.normal_fn(self, AHAx, *args, **kwargs)

            normal.normal_fn = new_normal_fn
            # Assume that none of the dims are the same anymore
            # Override this behavior for e.g. diagonal linops
            normal.oshape = tuple(d.next_unused(normal.ishape) for d in normal.oshape)
            normal._suffix += ".N"
            return normal
        pre = copy(self)
        pre.oshape = inner.ishape
        post = self.adjoint()  # Copy happens inside adjoint
        post.ishape = inner.oshape
        return post @ inner @ pre

    @staticmethod
    def split(linop, ibatch, obatch):
        """Return a split version of the linop such that`forward`
        performs a split version of the linop
        ibatch: tuple of slices of same length as ishape
        obatch: tuple of slices of same length as oshape
        """
        split = linop.split_forward(ibatch, obatch)
        return split

    @staticmethod
    def adj_split(linop, ibatch, obatch):
        """Split the adjoint version"""
        splitH = linop.adjoint().split_forward(obatch, ibatch).adjoint()
        return splitH

    @staticmethod
    def split_fn(linop, ibatch, obatch, /, **kwargs):
        """Return split versions of the data that can be passed
        into fn and adj_fn to produce split versions
        """
        return linop.split_forward_fn(ibatch, obatch, **kwargs)

    @staticmethod
    def adj_split_fn(linop, ibatch, obatch, /, **kwargs):
        return linop.split_forward_fn(obatch, ibatch, **kwargs)

    def flatten(self):
        """Get a flattened list of constituent linops for composition"""
        return [self]

    def compose(self, inner):
        """Do self AFTER inner"""
        before = inner.flatten()
        after = self.flatten()
        return torchlinops.Chain(*(after + before))

    def __add__(self, right):
        return torchlinops.Add(self, right)

    def __radd__(self, left):
        return torchlinops.Add(left, self)

    def __mul__(self, right):
        if isinstance(right, float) or isinstance(right, torch.Tensor):
            right = torchlinops.Scalar(weight=right)
            return self.compose(right)
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, float) or isinstance(left, torch.Tensor):
            left = torchlinops.Scalar(weight=left)
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

    def reset(self):
        """Clean up cached stuff."""
        self._adjoint = None
        self._normal = None

    # Pass these through to the shape representation
    @property
    def ishape(self):
        return self._shape.ishape

    @ishape.setter
    def ishape(self, val):
        self._shape.ishape = val

    @property
    def oshape(self):
        return self._shape.oshape

    @oshape.setter
    def oshape(self, val):
        self._shape.oshape = val

    def __copy__(self):
        """
        copying a linop:
        - Shares previous data
        - Removes references to adjoint and normal
        - Creates a new shape object, rather than using the old one
        """
        cls = type(self)
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)

        # Remove references to other objects
        new.reset()

        # Reset shape
        new._shape = deepcopy(self._shape)
        return new
