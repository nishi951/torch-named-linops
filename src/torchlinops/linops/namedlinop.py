import inspect
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
from torch.cuda import Stream

import torchlinops
import torchlinops.config as config
from torchlinops.nameddim import NamedDimension as ND, NamedShape, Shape
from torchlinops.utils import (
    INDENT,
    memory_aware_deepcopy,
    memory_aware_to,
)

__all__ = ["NamedLinop"]

logger = logging.getLogger("torchlinops")


def _log_transfer(msg):
    if config.log_device_transfers:
        logger.info(msg)


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
    stream : torch.cuda.Stream, optional
        CUDA stream to run the linop on. Ignored if the input is not CUDA.
    is_container : bool
        Set to ``True`` for linops whose primary function is to hold other linops.
    """

    is_container: bool = False
    """Set to True for linops whose primary function is to hold other linops."""

    def __init__(
        self,
        shape: NamedShape,
        name: Optional[str] = None,
        stream: Optional[Stream] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        shape : NamedShape
            The shape of this linop, e.g. ``NamedShape(("N",), ("M",))``
        name : str, optional
            Optional name to display for this linop.
        stream : torch.cuda.Stream, optional
            Stream to run the linop on. If the input is not CUDA, this is ignored.
        """
        super().__init__(**kwargs)
        # Note: this attribute is private because the `.shape` attribute may be derived
        # dynamically
        self._shape = shape
        self._suffix = ""
        self._name = name
        self.stream = stream
        self._setup()

    def _setup(self):
        """Helper method that should be called to reset the linop's state.
        Should be performed after any substantial changes to the linop."""
        self.reset_adjoint_and_normal()

    @final
    def forward(self, x: Tensor, context: Optional["SyncContext"] = None) -> Tensor:
        """Apply the forward operation $y = A(x)$.

        Do not override this method. Instead, override .fn() and .adj_fn().

        Parameters
        ----------
        x : Tensor
            Input tensor.
        context: SyncContext, optional
            Additional context for this linop's execution.
            Used for multi-gpu synchronization.

        Returns
        -------
        Tensor
            The result of applying this linop to *x*.
        """

        if "context" in inspect.signature(self.fn).parameters:
            context = SyncContext(
                linop=type(self),
                input_device=x.device,
                parent=context,
            )
            return self._run(x, context)
        return self._run(x)

    def _run(self, x, *args, **kwargs):
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                return self.fn(self, x, *args, **kwargs)
        return self.fn(self, x, *args, **kwargs)

    # Override
    @staticmethod
    def fn(linop, x: Tensor, /, context=None) -> Tensor:
        """Compute the forward operation $y = A(x)$.

        Override this in subclasses to define the linop's forward behavior.

        Parameters
        ----------
        linop : NamedLinop
            The linop instance (passed explicitly because this is a staticmethod).
        x : Tensor
            Input tensor.
        context : SyncContext, optional
            Execution context containing synchronization events for multi-GPU
            coordination. Only relevant for CUDA inputs.

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
    def adj_fn(linop, x: Tensor, /, context: Optional["SyncContext"] = None) -> Tensor:
        """Compute the adjoint operation $y = A^H(x)$.

        Override this in subclasses to define the linop's adjoint behavior.

        Parameters
        ----------
        linop : NamedLinop
            The linop instance.
        x : Tensor
            Input tensor.
        context : SyncContext, optional
            Execution context containing synchronization events for multi-GPU
            coordination. Only relevant for CUDA inputs.

        Returns
        -------
        Tensor
            Result of applying the adjoint $A^H$ to *x*.
        """
        return x

    # Override
    @staticmethod
    def normal_fn(
        linop, x: Tensor, /, context: Optional["SyncContext"] = None
    ) -> Tensor:
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
        context : SyncContext, optional
            Execution context containing synchronization events for multi-GPU
            coordination. Only relevant for CUDA inputs.

        Returns
        -------
        Tensor
            Result of applying $A^H A$ to *x*.
        """
        if "context" in inspect.signature(linop.fn).parameters:
            return linop.adj_fn(
                linop,
                linop.fn(linop, x, context=context),
                context=context,
            )

        return linop.adj_fn(linop, linop.fn(linop, x))

    @staticmethod
    def split(linop, tile: Mapping[ND | str, slice]) -> "NamedLinop":
        """Split a linop into a sub-linop for a given tile.

        Override this in subclasses to define how the linop decomposes when tiled
        along its named dimensions.

        Parameters
        ----------
        linop : NamedLinop
            The linop to split.
        tile : Mapping[ND | str, slice]
            Dictionary mapping dimension names to slices.

        Returns
        -------
        NamedLinop
            A new linop that operates on the specified slice of the data.
        """
        return type(linop)(linop._shape)

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
        adj = linop.adjoint()
        return type(adj).split(adj, tile).adjoint()

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
        try:
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
        except AttributeError as e:
            raise RuntimeError(f"AttributeError in {type(self).__name__}.H: {e}") from e

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
        try:
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
        except AttributeError as e:
            raise RuntimeError(f"AttributeError in {type(self).__name__}.N: {e}") from e

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
        if isinstance(right, (int, float)) or isinstance(right, torch.Tensor):
            right = torchlinops.Scalar(weight=right, ioshape=self.ishape)
            return self.compose(right)
        return NotImplemented

    def __rmul__(self, left) -> "NamedLinop":
        if isinstance(left, (int, float)) or isinstance(left, torch.Tensor):
            left = torchlinops.Scalar(weight=left, ioshape=self.oshape)
            return left.compose(self)
        return NotImplemented

    def __neg__(self) -> "NamedLinop":
        return (-1) * self

    def __sub__(self, right) -> "NamedLinop":
        return torchlinops.Add(self, -right)

    def __rsub__(self, left) -> "NamedLinop":
        if isinstance(left, NamedLinop):
            return torchlinops.Add(left, -self)
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
        out = f"{self.repr_name}({self.ishape} -> {self.oshape})"
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
            also calls ``.to()``. Will be deprecated along with cache_adjoint_normal.

        Returns
        -------
        NamedLinop
            The linop on the target device.
        """

        if config.cache_adjoint_normal:  # pragma: no cover
            config._warn_if_caching_enabled()
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

        # Create new shape
        new._shape = deepcopy(self._shape)
        new._setup()
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


@dataclass
class SyncContext:
    """Holds execution metadata for linop calls, propagated hierarchically through the call stack.

    Created fresh on each ``forward()`` call. Carries a ``start_event`` that
    marks "everything before this point is done" for CUDA synchronization.

    Container linops (``is_container = True``) record a CUDA event on the input
    device's current stream. Their child linops reuse this event rather than
    recording new ones, ensuring all children of a parallel container start from
    the same synchronization point.

    Non-container linops that are children of non-containers record their own
    fresh event.

    Attributes
    ----------
    linop : type[NamedLinop]
        Class of the linop that owns this context.
    input_device : torch.device
        Device of the input tensor to this linop.
    parent : SyncContext, optional
        Calling context from the parent linop. ``None`` at the top level.
    start_event : torch.cuda.Event, optional
        Synchronization event. Reused from parent if parent is a container;
        otherwise recorded fresh on the current stream of the input device.
    """

    linop: type[NamedLinop]
    """Class of linop that owns this context"""
    input_device: torch.device
    """Device of input to this linop"""
    parent: Optional["SyncContext"] = None
    """Calling context."""
    start_event: torch.cuda.Event | None = None
    """Automatically propagate from parent if parent is parallelizable.
    Otherwise, record an event on the current stream of the input device"""

    def __post_init__(self):
        if self.input_device.type == "cuda":
            if (
                self.parent is not None
                and self.parent.linop.is_container
                and self.parent.start_event is not None
            ):
                logger.debug(
                    f"{self.linop.__name__} with parent {None if self.parent is None else self.parent.linop.__name__} reusing event {self.parent.start_event}"
                )
                self.start_event = self.parent.start_event
            if self.start_event is None and self.linop.is_container:
                self.start_event = torch.cuda.current_stream(
                    self.input_device
                ).record_event()
                logger.debug(
                    f"{self.linop.__name__} with parent {None if self.parent is None else self.parent.linop.__name__} recorded event {self.start_event} on {self.input_device} stream {torch.cuda.current_stream(self.input_device)}"
                )
