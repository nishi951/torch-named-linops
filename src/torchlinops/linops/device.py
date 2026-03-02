from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional
from warnings import warn

import torch
from torch.cuda import Event, Stream, default_stream

from torchlinops.utils import INDENT, RepeatedEvent, default_to

from ..nameddim import NamedShape as NS, Shape
from .identity import Identity
from .namedlinop import NamedLinop, ForwardedAttribute

__all__ = ["ToDevice"]

# Registry to keep track of transfer streams already created.
_TRANSFER_STREAMS_REGISTRY = {}


@dataclass
class DeviceSpec:
    """Lightweight data structure for holding useful CUDA-related objects for multi-GPU computation."""

    device: Any = field(default_factory=lambda: torch.device("cpu"))
    """Device for the streams."""
    compute_stream: Optional[Stream] = None
    """Stream used for computation."""
    transfer_stream: Optional[Stream] = None
    """Stream used for data transfer."""

    def __post_init__(self):
        """Ensure self.device is a proper torch.device."""
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    def p2p_setup(self, other_device):
        """Sets up compute and transfer streams for peer2peer transfers, if not set yet."""
        if self.device.type == "cuda" and other_device.type == "cuda":
            if self.compute_stream is None:
                self.compute_stream = default_stream(self.device)
            if self.transfer_stream is None:
                self.transfer_stream = self.get_transfer_stream(
                    self.device, other_device
                )

    @property
    def type(self):
        """Passthrough for torch.device.type."""
        return self.device.type

    @staticmethod
    def get_transfer_stream(source_device: torch.device, target_device: torch.device):
        """Return the stream used for device transfers associated with this device.

        TODO: For now, we limit to one transfer stream per device per runtime. In the future,
        we may want one stream per source/target device pair.
        """
        if (source_device, target_device) in _TRANSFER_STREAMS_REGISTRY:
            return _TRANSFER_STREAMS_REGISTRY[(source_device, target_device)]
        # Create a new stream
        new_stream = Stream(source_device)
        _TRANSFER_STREAMS_REGISTRY[(source_device, target_device)] = new_stream
        return new_stream


class ToDevice(NamedLinop):
    """Transfer tensors between devices as a named linear operator.

    The forward operation moves a tensor from ``idevice`` to ``odevice``.
    The adjoint reverses the direction. The normal $T^H T$ is the identity
    (device round-trip is lossless).

    For CUDA-to-CUDA transfers, streams and events are used for asynchronous
    pipelined execution.

    Attributes
    ----------
    ispec : DeviceSpec
        Source (input) device specification.
    ospec : DeviceSpec
        Target (output) device specification.
    input_linop : List[NamedLinop]
        The linop to wait for if gpu2gpu transfers are necessary.
        It is a list with length 1 to prevent registering the linop as a submodule.
    input_event_key : str
        The name of the attribute on input_linop containing the triggering event.
    """

    def __init__(
        self,
        idevice: DeviceSpec | torch.device | None,
        odevice: DeviceSpec | torch.device | None,
        ioshape: Optional[Shape] = None,
    ):
        """
        Parameters
        ----------
        idevice_spec : DeviceSpec | None
            Source (input) device specification.
        odevice_spec : DeviceSpec | None
            Target (output) device specification.
        ioshape : Shape, optional
            Named dimensions (same for input and output since this is diagonal).
        input_ready_event : Event, optional
            A CUDA event to wait on before initiating the transfer.
        """
        super().__init__(NS(ioshape))

        idevice = default_to(torch.device("cpu"), idevice)
        odevice = default_to(torch.device("cpu"), odevice)
        if not isinstance(idevice, DeviceSpec):
            self.ispec = DeviceSpec(idevice)
        else:
            self.ispec = idevice
        if not isinstance(odevice, DeviceSpec):
            self.ospec = DeviceSpec(odevice)
        else:
            self.ospec = odevice

        # Perform any necessary setup for data transfer between these devices.
        self.ispec.p2p_setup(self.ospec.device)
        self.ospec.p2p_setup(self.ispec.device)

        if self.ispec.device.type == "cuda" and self.ospec.device.type == "cuda":
            self.is_gpu2gpu = True
        else:
            self.is_gpu2gpu = False

        # Set up input event
        self._input_ready_event = ForwardedAttribute()

        # By default, link it to the start event
        self.input_ready_event = (self, "start_event")

    @property
    def input_ready_event(self):
        """Dynamically determine the event to wait for from a linop and attribute name.

        This event is necessary for gpu-gpu transfers.

        For example, in a chain like this:

        ToDevice @ A

        Set self.input_linop[0] = A and self.input_event_key = "end_event"
        so that we use A.end_event as the triggering event.

        However, if ToDevice occurs inside a composing linop that allows for
        parallel execution, e.g.

        C = Concat(
            Chain(ToDevice1, A, ...),
            Chain(ToDevice2, B, ...),
            ...
        )

        Then we may want to set

        ToDevice1.input_linop[0] = C
        ToDevice1.input_event_key = "start_event"
        ToDevice2.input_linop[0] = C
        ToDevice2.input_event_key = "start_event"

        So that both ToDevice linops trigger on the beginning of C.
        """
        return self._input_ready_event.value

    @input_ready_event.setter
    def input_ready_event(self, value):
        if isinstance(value, tuple):
            self._input_ready_event.forward_to(*value)
        else:
            self._input_ready_event = value

    @staticmethod
    def _fn(
        x,
        ispec: DeviceSpec,
        ospec: DeviceSpec,
        input_ready_event: Optional[Event] = None,
    ):
        idevice, odevice = ispec.device, ospec.device
        if x.device != idevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {idevice}"
            )

        # GPU -> GPU
        if idevice.type == "cuda" and odevice.type == "cuda":
            if input_ready_event is None:
                warn(
                    "Peer-to-peer device transfer with input_ready_event = None detected. Results may not be accurate."
                )
            return _gpu2gpu_transfer(
                x,
                odevice,
                ispec.transfer_stream,
                ospec.compute_stream,
                input_ready_event,
            )
        # CPU -> GPU, GPU -> CPU or CPU -> CPU
        return x.to(odevice, non_blocking=True)

    @staticmethod
    def fn(todevice, x, /):
        return todevice._fn(
            x,
            todevice.ispec,
            todevice.ospec,
            todevice.input_ready_event,
        )

    @staticmethod
    def adj_fn(todevice, x, /):
        return todevice._fn(
            x,
            todevice.ospec,
            todevice.ispec,
            todevice.input_ready_event,
        )

    def adjoint(self):
        adj = copy(self)
        adj._shape = adj._shape.H
        adj.ispec, adj.ospec = self.ospec, self.ispec
        return adj

    def normal(self, inner=None):
        if inner is None:
            return Identity()
        return super().normal(inner)

    def split_forward(self, ibatch, obatch):
        """Return a new instance"""
        return copy(self)

    def __repr__(self):
        """Helps prevent recursion error caused by .H and .N"""
        if (
            self.ispec.compute_stream is not None
            or self.ispec.transfer_stream is not None
        ):
            irepr = f"{self.ispec.device}, compute: 0x{self.ispec.compute_stream.cuda_stream:x}, transfer: 0x{self.ispec.transfer_stream.cuda_stream:x}"
        else:
            irepr = f"{self.ispec.device}"
        if (
            self.ospec.compute_stream is not None
            or self.ospec.transfer_stream is not None
        ):
            orepr = f"{self.ospec.device}, compute: 0x{self.ospec.compute_stream.cuda_stream:x}, transfer: 0x{self.ospec.transfer_stream.cuda_stream:x}"
        else:
            orepr = f"{self.ospec.device}"
        if self.input_ready_event is not None:
            input_ready_event_repr = f"on: {self.input_ready_event.event_id:x}"
        else:
            input_ready_event_repr = ""
        out = f"({input_ready_event_repr} | {irepr} -> {orepr})"
        out = INDENT.indent(out)
        return out


def _gpu2gpu_transfer(x, odevice, transfer_stream, target_stream, input_ready_event):
    """Perform efficient gpu-gpu transfer with a dedicated transfer stream and event-based triggering.

    Parameters
    ----------
    x : Tensor
        The torch tensor to be transferred.
    odevice : torch.device
        The target device.
    transfer_stream : Stream
        The stream on the source device on which to queue the transfer.
    target_stream : Stream
        The stream on the target device that needs `x` for computation.
    input_ready_event : Event
        The CUDA event that the transfer stream should wait for before initiating transfer.
        Allows fine-grained control of transfer timing. The event should be queued to
        record() when x is ready to be transferred.

    Returns
    -------
    Tensor
        The tensor, on the target device.
    """
    # with torch.cuda.stream(transfer_stream):
    with transfer_stream:
        if input_ready_event is not None:
            # if isinstance(input_ready_event, RepeatedEvent):
            #     transfer_stream.wait_event(input_ready_event.last_event)
            # else:
            transfer_stream.wait_event(input_ready_event)
        out = x.to(odevice, non_blocking=True)
    # Don't mess with x's memory until transfer is completed
    x.record_stream(transfer_stream)
    # Target stream should wait until transfer is complete
    target_stream.wait_stream(transfer_stream)
    return out
