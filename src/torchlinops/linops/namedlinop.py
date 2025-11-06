import logging
import traceback
import types
from collections import defaultdict
from collections.abc import Mapping
from copy import copy, deepcopy
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda import Event, Stream

import torchlinops
import torchlinops.config as config
from torchlinops.utils import (
    INDENT,
    check_signature,
    memory_aware_deepcopy,
    memory_aware_to,
)
from .nameddim import NS
from .nameddim import NamedDimension as ND
from .nameddim import NamedShape, NDorStr

__all__ = ["NamedLinop"]

logger = logging.getLogger("torchlinops")


class NamedLinop(nn.Module):
    """Base class for all NamedLinops"""

    def __init__(
        self,
        shape: NamedShape,
        name: Optional[str] = None,
        stream: Optional[Stream] = None,
        start_event: Optional[Event] = None,
    ):
        """
        Parameters
        ----------
        shape : NamedShape
            The shape of this linop, e.g. ``NS(("N",), ("M",))``
        name : str, optional
            Optional name to display for this linop.
        stream : Stream, optional
            The CUDA stream on which to run this linop.
        start_event : Event, optional
            An event that signals when the linop has started. Useful for synchronizing multiple
            linops across multiple devices.
        """
        super().__init__()
        self._shape = shape

        self.reset_adjoint_and_normal()

        self._suffix = ""
        self._name = name
        self.stream = stream
        self.start_event = start_event

    def forward(self, x: Tensor) -> Tensor:
        if self.start_event is not None:
            self.start_event.record()
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                y = self.fn(self, x)
            x.record_stream(self.stream)
            return y
        return self.fn(self, x)

    def apply(self, x: Tensor) -> Tensor:
        """Apply the linear operator to a tensor."""
        return LinopFunction.apply(x, self)

    # Override
    @staticmethod
    def fn(linop, x: Tensor, /) -> Tensor:
        """Apply the linop to a tensor.

        Parameters
        ----------
        x : Tensor
            The input to the linop.

        Returns
        -------
        Tensor
            A(x)
            The result of applying the linop.

        Notes
        -----
        - Other parameters are passed in as attributes of the linop object.
        - Declared as a staticmethod because it needs to be unbound when swapping with adj_fn.
        """
        return x

    # Override
    @staticmethod
    def adj_fn(linop, x: Tensor, /) -> Tensor:
        """Apply the adjoint of a linop to a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input to the linop.

        Returns
        -------
        Tensor
            A.H(x)
            The result of applying the linop's adjoint.
        """
        return x

    # Override
    @staticmethod
    def normal_fn(linop, x: Tensor, /) -> Tensor:
        """Apply the normal of a linop to a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input to the linop.

        Returns
        -------
        Tensor
            A.H(A(x)) == A.N(x)
            The result of applying the linop's normal function.
        """
        return linop.adj_fn(linop, linop.fn(linop, x))

    # Override
    def split_forward(self, ibatch, obatch):
        """Split this linop into a sub-linop according to slices over its dimensions

        Parameters
        ----------
        ibatch : tuple[slice, ...]
            The slices over the input dimensions.
        obatch : tuple[slice, ...]
            The slices over the output dimensions.
        """

        return type(self)(self._shape)

    # Override
    # TODO: Deprecate
    def split_forward_fn(self, ibatch, obatch, /, data=None):
        """Split this linop's data."""
        return None

    # Override
    def size(self, dim: str) -> int | None:
        """Get the size of a particular dim, or return
        None if this linop doesn't determine the size
        """
        return None

    # Override
    # TODO: Deprecate
    def size_fn(self, dim: str, /, data=None):
        """Functional version of size. Determines sizes from kwargs
        kwargs should be the same as the inputs to fn or adj_fn
        Return None if this linop doesn't determine the size of dim
        """
        return None

    # Probably don't override these
    @property
    def dims(self) -> set:
        """Get the set of dims that appear in this linop."""
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
        adj._update_suffix(adjoint=True)
        return adj

    def _update_suffix(self, adjoint: bool = False, normal: bool = False):
        if adjoint:
            if self._suffix.endswith(".H"):
                self._suffix = self._suffix[:-2]
            else:
                self._suffix += ".H"
        elif normal:
            self._suffix += ".N"

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
        if config.inner_not_relevant(inner):
            normal = copy(self)
            normal._shape = self._shape.N

            # Auxiliary object
            # Avoids creating lambda functions, which enables multiprocessing
            function_table = NormalFunctionLookup(self)
            # Static
            normal.fn = function_table.new_forward_adjoint_fn
            normal.adj_fn = function_table.new_forward_adjoint_fn
            normal.normal_fn = function_table.new_normal_fn
            # Bind `self` with partial to avoid weird multiprocessing-only error?
            normal.adjoint = partial(new_normal_adjoint, self=normal)
            # normal.adjoint = new_normal_adjoint.__get__(normal) # This one doesn't work

            # Assume that none of the dims are the same anymore
            # Override this behavior for e.g. diagonal linops
            normal.oshape = tuple(d.next_unused(normal.ishape) for d in normal.oshape)
            # Remember which shapes were updated
            normal._shape_updates = {
                d: d.next_unused(normal.ishape) for d in normal.oshape
            }
            normal._update_suffix(normal=True)
            return normal
        pre = copy(self)
        pre.oshape = inner.ishape
        post = self.adjoint()  # Copy happens inside adjoint
        post.ishape = inner.oshape
        normal = post @ inner @ pre
        normal._shape_updates = getattr(inner, "_shape_updates", {})
        return normal

    @staticmethod
    def split(linop, tile: Mapping[ND | str, slice]):
        """Split a linop into sub-linops.

        Parameters
        ----------
        linop : NamedLinop
            The linop to split.
        tile : Mapping[ND | str, slice]
            Dictionary specifying how to slice the linop dimensions
        """
        ibatch = [tile.get(dim, slice(None)) for dim in linop.ishape]
        obatch = [tile.get(dim, slice(None)) for dim in linop.oshape]
        return linop.split_forward(ibatch, obatch)

    @staticmethod
    def adj_split(linop, tile: Mapping[ND | str, slice]):
        """Split the adjoint version"""
        ibatch = [tile.get(dim, slice(None)) for dim in linop.ishape]
        obatch = [tile.get(dim, slice(None)) for dim in linop.oshape]
        splitH = linop.adjoint().split_forward(obatch, ibatch).adjoint()
        return splitH

    def flatten(self):
        """Get a flattened list of constituent linops for composition."""
        return [self]

    def compose(self, inner):
        """Compose this linop with another linop.

        Parameters
        ----------
        inner : NamedLinop
            The linop to call before this one.

        Returns
        -------
        NamedLinop
            The composition of self and inner. If A = self and B = inner then this returns
            C = AB.
        """
        before = inner.flatten()
        after = self.flatten()
        return torchlinops.Chain(*(before + after))

    def __add__(self, right):
        return torchlinops.Add(self, right)

    def __radd__(self, left):
        return torchlinops.Add(left, self)

    def __mul__(self, right):
        if isinstance(right, float) or isinstance(right, torch.Tensor):
            right = torchlinops.Scalar(weight=right, ioshape=self.ishape)
            return self.compose(right)
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, float) or isinstance(left, torch.Tensor):
            left = torchlinops.Scalar(weight=left, ioshape=self.oshape)
            return left.compose(self)
        return NotImplemented

    def __matmul__(self, right):
        if isinstance(right, NamedLinop):
            return self.compose(right)
        if isinstance(right, torch.Tensor):
            return self(right)
        return NotImplemented

    def __rmatmul__(self, left):
        return left.compose(self)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        return type(self).__name__

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def repr_name(self):
        return self.name + self._suffix

    def __repr__(self):
        event = ""
        if self.start_event is not None:
            event = repr(self.start_event)
        out = f"{self.repr_name}({event}{self.ishape} -> {self.oshape})"
        out = INDENT.indent(out)
        return out

    def reset_adjoint_and_normal(self):
        self._adjoint = None
        self._normal = None

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        self._shape = val

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

    def to(self, device, memory_aware: bool = False, called_by_adjoint: bool = False):
        if self._adjoint and not called_by_adjoint:
            # bool flag avoids infinite recursion
            self._adjoint[0] = self._adjoint[0].to(
                device, memory_aware, called_by_adjoint=True
            )
        if self._normal:
            self._normal[0] = self._normal[0].to(device, memory_aware)
        if memory_aware:
            return memory_aware_to(self, device)
        return super().to(device)

    def __copy__(self):
        """Specialized copying for linops.

        Notes
        -----
        - Shares previous data
        - Removes references to adjoint and normal
        - Creates a new shape object, rather than using the old one
        """
        cls = type(self)
        new = cls.__new__(cls)
        new.__dict__ = self.__dict__.copy()
        # Pytorch-specific module state dictionaries
        # Mirror those used in `__getattr__``
        # See https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/module.py#L1915
        new._parameters = new._parameters.copy()
        new._modules = new._modules.copy()
        new._buffers = new._buffers.copy()

        # Remove references to other objects
        new.reset_adjoint_and_normal()

        # Create new shape
        new._shape = deepcopy(self._shape)
        return new

    def __deepcopy__(self, _):
        return memory_aware_deepcopy(self)


class NormalFunctionLookup:
    """Helper class for creating new normal functions for a normal linop.

    If the linop is A, and its normal is A.N, this helps with computing A.N.H and A.N.N. Note that A.N.H = A.N

    Helps with multiprocessing by avoiding lambda function definitions, thereby maintaining pickleability.
    """

    def __init__(self, linop):
        self.linop = linop

    def new_forward_adjoint_fn(self, _, x, *args, **kwargs):
        """Replace forward/adjoint with normal fn from self."""
        return self.linop.normal_fn(self.linop, x, *args, **kwargs)

    def new_normal_fn(self, _, x, *args, **kwargs):
        """Replace normal fn with normal_fn(normal_fn(  ))"""
        AHAx = self.linop.normal_fn(self.linop, x, *args, **kwargs)
        return self.linop.normal_fn(self.linop, AHAx, *args, **kwargs)


def new_normal_adjoint(self):
    """Adjoint-of-normal creation helper function.

    Top-level definition to maintain pickleability.
    """
    adj = copy(self)
    adj._shape = adj._shape.H
    return adj


class LinopFunction(torch.autograd.Function):
    """Wrap a linop in an autograd function.

    At one point, this may have helped with memory usage in some cases.

    Experimental.
    """

    @staticmethod
    def forward(input_, linop):
        return linop(input_)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input_, linop = inputs
        ctx.linop = linop
        ctx.input_shape = input_.shape

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_linop = None
        linop = ctx.linop
        input_shape = ctx.input_shape
        grad_input = linop.H(grad_output)
        grad_input = torch.broadcast_to(grad_input, input_shape)
        return grad_input, grad_linop
