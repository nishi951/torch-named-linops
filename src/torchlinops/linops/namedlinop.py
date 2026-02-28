import logging
import traceback
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda import Event, Stream, default_stream

import torchlinops
import torchlinops.config as config
from torchlinops.nameddim import NamedDimension as ND, NamedShape, Shape
from torchlinops.utils import (
    INDENT,
    RepeatedEvent,
    default_to,
    memory_aware_deepcopy,
    memory_aware_to,
)

__all__ = ["NamedLinop"]

logger = logging.getLogger("torchlinops")


class NamedLinop(nn.Module):
    """Base class for all named linear operators.

        A ``NamedLinop`` represents a linear map $A : X \\to Y$ where the input and
        output tensor dimensions are identified by name (e.g. ``("Nx", "Ny") -> ("Kx", "Ky")``).

        Subclass this to implement concrete operators. At minimum, override ``fn``
        and ``adj_fn`` as static methods.

        Attributes
        ----------
        shape : NamedShape
            The named shape of the linop, containing ``ishape`` and ``oshape``.
        stream : torch.cuda.Stream
            Optional cuda Stream to run this linop on.
        start_event : Event, optional
            An event that signals when the linop has started. Useful for synchronizing
            multiple linops across multiple devices.
        end_event : Event, optional
            An event that signals when the linop has completed. Useful for synchronizing multiple

    linops across multiple devices.
    """

    def __init__(
        self,
        shape: NamedShape,
        name: Optional[str] = None,
        stream: Optional[Stream] = None,
        start_event: Optional[Event] = None,
        end_event: Optional[Event] = None,
    ):
        """
        Parameters
        ----------
        shape : NamedShape
            The shape of this linop, e.g. ``NamedShape(("N",), ("M",))``
        name : str, optional
            Optional name to display for this linop.
        stream : torch.cuda.Stream
            Optional cuda Stream to run this linop on.
        start_event : Event, optional
            An event that signals when the linop has started. Useful for synchronizing multiple
            linops across multiple devices.
        end_event : Event, optional
            An event that signals when the linop has completed. Useful for synchronizing multiple
            linops across multiple devices.
        """
        super().__init__()
        # Note: this attribute is private because the `.shape` attribute may be derived
        # dynamically
        self._shape = shape

        self.reset_adjoint_and_normal()

        self._suffix = ""
        self._name = name
        self.stream = stream
        self._start_event = ForwardedAttribute()
        self._end_event = ForwardedAttribute()

    @final
    def forward(self, x: Tensor) -> Tensor:
        """Apply the forward operation $y = A(x)$.

        If a CUDA stream is assigned, execution is dispatched to that stream.
        If a ``start_event`` is set, it is recorded before execution begins,
        allowing other operators to synchronize on it.

        Do not override this method. Instead, override .fn() and .adj_fn().

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            The result of applying this linop to *x*.
        """
        if x.is_cuda:
            stream = default_to(default_stream(x.device), self.stream)
            if self.start_event is None:
                self.start_event = stream.record_event()
            with stream:
                y = self.fn(self, x)
            x.record_stream(stream)
            if self.end_event is None:
                self.end_event = stream.record_event()
        else:
            y = self.fn(self, x)
        return y

    def apply(self, x: Tensor) -> Tensor:
        """Apply the linear operator to a tensor."""
        return LinopFunction.apply(x, self)

    # Override
    @staticmethod
    def fn(linop, x: Tensor, /) -> Tensor:
        """Compute the forward operation $y = A(x)$.

        Override this in subclasses to define the linop's forward behavior.

        Parameters
        ----------
        linop : NamedLinop
            The linop instance (passed explicitly because this is a staticmethod).
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Result of applying the linop to *x*.

        Notes
        -----
        Declared as a staticmethod so that ``adjoint()`` can swap ``fn`` and
        ``adj_fn`` on a shallow copy without bound-method complications.
        """
        return x

    # Override
    @staticmethod
    def adj_fn(linop, x: Tensor, /) -> Tensor:
        """Compute the adjoint operation $y = A^H(x)$.

        Override this in subclasses to define the linop's adjoint behavior.

        Parameters
        ----------
        linop : NamedLinop
            The linop instance.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Result of applying the adjoint $A^H$ to *x*.
        """
        return x

    # Override
    @staticmethod
    def normal_fn(linop, x: Tensor, /) -> Tensor:
        """Compute the normal operation $y = A^H A(x)$.

        The default implementation composes ``adj_fn(fn(x))``. Override this
        in subclasses that have an efficient closed-form normal (e.g.
        ``Diagonal``, ``FFT``).

        Parameters
        ----------
        linop : NamedLinop
            The linop instance.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Result of applying $A^H A$ to *x*.
        """
        return linop.adj_fn(linop, linop.fn(linop, x))

    # Override
    def split_forward(self, ibatch, obatch) -> "NamedLinop":
        """Split this linop into a sub-linop according to slices over its dimensions.

        Override this in subclasses to define how the linop decomposes when tiled
        along its named dimensions.

        Parameters
        ----------
        ibatch : tuple[slice, ...]
            Slices over the input dimensions, one per element of ``ishape``.
        obatch : tuple[slice, ...]
            Slices over the output dimensions, one per element of ``oshape``.

        Returns
        -------
        NamedLinop
            A new linop that operates on the specified slice of the data.
        """

        return type(self)(self._shape)

    # Override
    def size(self, dim: str) -> int | None:
        """Return the concrete size of *dim*, or ``None`` if this linop does not determine it.

        Parameters
        ----------
        dim : str
            The named dimension to query.

        Returns
        -------
        int or None
            The size of the dimension, or ``None``.
        """
        return None

    @final
    @property
    def dims(self) -> set:
        """Get the set of dims that appear in this linop."""
        return set(self.ishape).union(set(self.oshape))

    @final
    @property
    def H(self) -> "NamedLinop":
        """Adjoint operator $A^H$.

        By default, creates a new adjoint on each access. Set
        ``torchlinops.config.cache_adjoint_normal = True`` to enable caching
        (deprecated).
        """
        if config.cache_adjoint_normal:
            config._warn_if_caching_enabled()
            if self._adjoint is None:
                try:
                    _adjoint = self.adjoint()
                    _adjoint._adjoint = [self]
                    self._adjoint = [_adjoint]
                except AttributeError as e:
                    traceback.print_exc()
                    raise e
                logger.debug(
                    f"{type(self).__name__}: Making new adjoint {_adjoint._shape}"
                )
            return self._adjoint[0]
        return self.adjoint()

    def adjoint(self) -> "NamedLinop":
        """Create the adjoint operator $A^H$.

        The default implementation shallow-copies this linop, swaps ``fn`` and
        ``adj_fn``, and flips the shape. Override this in subclasses that need
        special adjoint construction (e.g. conjugating weights).

        Returns
        -------
        NamedLinop
            The adjoint operator, sharing the same underlying data.
        """
        adj = copy(self)  # Retains data
        adj._shape = adj._shape.H
        # Swap functions (requires staticmethod)
        adj.fn, adj.adj_fn = adj.adj_fn, adj.fn
        adj.split, adj.adj_split = adj.adj_split, adj.split
        adj._update_suffix(adjoint=True)
        return adj

    @final
    def _update_suffix(self, adjoint: bool = False, normal: bool = False):
        if adjoint:
            if self._suffix.endswith(".H"):
                self._suffix = self._suffix[:-2]
            else:
                self._suffix += ".H"
        elif normal:
            self._suffix += ".N"

    @final
    @property
    def N(self) -> "NamedLinop":
        """Normal operator $A^H A$.

        Note that the naive normal operator can always be created via ``A.H @ A``.
        This function is reserved for custom behavior, as many linops have
        optimized normal forms.

        By default, creates a new normal on each access. Set
        ``torchlinops.config.cache_adjoint_normal = True`` to enable caching
        (deprecated).
        """
        if config.cache_adjoint_normal:
            config._warn_if_caching_enabled()
            if self._normal is None:
                try:
                    _normal = self.normal()
                    self._normal = [_normal]
                except AttributeError as e:
                    traceback.print_exc()
                    raise e
            return self._normal[0]
        return self.normal()

    def normal(self, inner=None) -> "NamedLinop":
        """Create the normal operator $A^H A$, optionally with an inner operator.

        When *inner* is ``None`` (or ``Identity`` with the reduce-identity config
        enabled), creates a linop whose forward pass calls ``normal_fn``.

        When *inner* is provided, constructs the composition $A^H \\cdot \\text{inner} \\cdot A$,
        which is used for Toeplitz embedding and similar optimizations.

        Parameters
        ----------
        inner : NamedLinop, optional
            An optional inner operator for Toeplitz embedding. If ``None``,
            the standard normal $A^H A$ is computed.

        Returns
        -------
        NamedLinop
            The normal operator.
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

    @final
    @staticmethod
    def split(linop, tile: Mapping[ND | str, slice]) -> "NamedLinop":
        """Split a linop into a sub-linop for a given tile.

        Translates a tile dictionary into per-dimension slices and delegates
        to ``split_forward``.

        Parameters
        ----------
        linop : NamedLinop
            The linop to split.
        tile : Mapping[ND | str, slice]
            Dictionary mapping dimension names to slices.

        Returns
        -------
        NamedLinop
            The sub-linop operating on the specified tile.
        """
        ibatch = [tile.get(dim, slice(None)) for dim in linop.ishape]
        obatch = [tile.get(dim, slice(None)) for dim in linop.oshape]
        return linop.split_forward(ibatch, obatch)

    @final
    @staticmethod
    def adj_split(linop, tile: Mapping[ND | str, slice]) -> "NamedLinop":
        """Split the adjoint of this linop for a given tile.

        Constructs the adjoint, splits it according to *tile*, and returns the
        adjoint of the split.

        Parameters
        ----------
        linop : NamedLinop
            The linop whose adjoint should be split.
        tile : Mapping[ND | str, slice]
            Dictionary mapping dimension names to slices.

        Returns
        -------
        NamedLinop
            The split adjoint sub-linop.
        """
        ibatch = [tile.get(dim, slice(None)) for dim in linop.ishape]
        obatch = [tile.get(dim, slice(None)) for dim in linop.oshape]
        splitH = linop.adjoint().split_forward(obatch, ibatch).adjoint()
        return splitH

    @final
    def flatten(self) -> list["NamedLinop"]:
        """Get a flattened list of constituent linops for composition."""
        return [self]

    def compose(self, inner) -> "NamedLinop":
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

    def __add__(self, right) -> "NamedLinop":
        return torchlinops.Add(self, right)

    def __radd__(self, left) -> "NamedLinop":
        return torchlinops.Add(left, self)

    def __mul__(self, right) -> "NamedLinop":
        if isinstance(right, float) or isinstance(right, torch.Tensor):
            right = torchlinops.Scalar(weight=right, ioshape=self.ishape)
            return self.compose(right)
        return NotImplemented

    def __rmul__(self, left) -> "NamedLinop":
        if isinstance(left, float) or isinstance(left, torch.Tensor):
            left = torchlinops.Scalar(weight=left, ioshape=self.oshape)
            return left.compose(self)
        return NotImplemented

    def __matmul__(self, right) -> "NamedLinop":
        if isinstance(right, NamedLinop):
            return self.compose(right)
        if isinstance(right, torch.Tensor):
            return self(right)
        return NotImplemented

    def __rmatmul__(self, left) -> "NamedLinop":
        if not isinstance(left, NamedLinop):
            raise ValueError(
                f"__rmatmul__ of linop {type(self)} with non-linop of type {type(left)} is undefined."
            )
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
    def shape(self) -> Shape:
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
        """Move this linop (and its cached adjoint/normal) to *device*.

        Parameters
        ----------
        device : torch.device or str
            Target device.
        memory_aware : bool, default False
            If ``True``, use ``memory_aware_to`` which preserves shared-storage
            topology when moving tensors.
        called_by_adjoint : bool, default False
            Internal flag to prevent infinite recursion when the adjoint
            also calls ``.to()``.

        Returns
        -------
        NamedLinop
            The linop on the target device.
        """
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

    @property
    def start_event(self):
        return self._start_event.value

    @start_event.setter
    def start_event(self, event):
        """
        Parameters
        ----------
        event : Event | tuple[Any, str]
            If a bare Event is provided, use that event on this object.
            If a tuple, interpret it as a reference to forward. e.g. event = (other_linop, 'start_event')
            will forward this linop's linop.start_event to other_linop.start_event
        """
        if isinstance(event, tuple):
            self._start_event.forward_to(*event)
        else:
            self._start_event.value = event

    @property
    def end_event(self):
        return self._end_event.value

    @end_event.setter
    def end_event(self, event):
        if isinstance(event, tuple):
            self._end_event.forward_to(*event)
        else:
            self._end_event.value = event

    @final
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

    @final
    def __deepcopy__(self, _):
        return memory_aware_deepcopy(self)


class NormalFunctionLookup:
    """Function table for the normal operator $A^N = A^H A$.

    Provides named methods that serve as ``fn``, ``adj_fn``, and ``normal_fn``
    for a normal-operator linop. Using a class instead of lambdas keeps the
    linop picklable (required for ``torch.multiprocessing``).

    Since $A^N$ is self-adjoint, its forward and adjoint are the same function.
    Its own normal is $(A^H A)^2$.
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


@dataclass
class ForwardedAttribute:
    """Special dataclass for forwarding attribute access to other objects.

    TODO: Make this a descriptor somehow?

    Works best when combined with a @property.

    Example:

    class MyClass:
        def __init__(self):
            self._foo = ForwardedAttribute()
        @property
        def foo(self):
            return self._foo.value

        @foo.setter
        def foo(self, new_value):
            if isinstance(new_value, tuple):
                self._foo.forward_to(*new_value)
            else:
                self._foo = new_value

    """

    allow_set_upstream: bool = True
    """If true, allow setting this value to affect upstream values."""
    _value: Optional[Any] = None
    _obj: Optional[Any] = None
    _attr: Optional[Any] = None

    @property
    def value(self):
        if self._obj is None:
            # No object to forward to
            return self._value
        elif self._value is not None:
            # A preset value overrides forwarded reference
            return self._value
        return getattr(self._obj, self._attr)

    @value.setter
    def value(self, new_value):
        if self._obj is None:
            # No object to forward to
            self._value = new_value
        elif self.allow_set_upstream:
            setattr(self._obj, self._attr, new_value)
        else:
            # Don't overwrite upstream
            self._value = new_value

            # Clear pointer
            self._obj = None
            self._attr = None

    def forward_to(self, obj, attr):
        """Create reference"""
        self._obj = obj
        self._attr = attr
        self._value = None  # Reset value

    @property
    def is_forwarded(self) -> bool:
        return self._obj is not None
