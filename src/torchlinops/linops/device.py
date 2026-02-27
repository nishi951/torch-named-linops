from copy import copy
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional
from warnings import warn

import torch
from torch.cuda import Event, Stream, default_stream

from torchlinops.utils import INDENT, RepeatedEvent, default_to

from ..nameddim import NamedShape as NS, Shape
from .identity import Identity
from .namedlinop import NamedLinop

__all__ = ["ToDevice"]


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
                self.transfer_stream = self.get_transfer_stream(self.device)

    @property
    def type(self):
        """Passthrough for torch.device.type."""
        return self.device.type

    @staticmethod
    def get_transfer_stream(device: torch.device):
        """Return the stream used for device transfers associated with this device."""


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
    wait_event : Event or None
        Optional event to wait on before starting the transfer.
    """

    def __init__(
        self,
        idevice: DeviceSpec | torch.device | None,
        odevice: DeviceSpec | torch.device | None,
        ioshape: Optional[Shape] = None,
        wait_event: Optional[Event] = None,
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
        wait_event : Event, optional
            A CUDA event to wait on before initiating the transfer.
        """
        super().__init__(NS(ioshape))

        idevice = default_to(torch.device("cpu"), idevice)
        odevice = default_to(torch.device("cpu"), odevice)
        if not isinstance(idevice, DeviceSpec):
            self.ispec = DeviceSpec(idevice)
        if not isinstance(odevice, DeviceSpec):
            self.ospec = DeviceSpec(odevice)

        self.ispec.p2p_setup(self.ospec.device)
        self.ospec.p2p_setup(self.ispec.device)

        if self.ispec.device.type == "cuda" and self.ospec.device.type == "cuda":
            # Only initialize peer-to-peer access if both devices are cuda.
            self.wait_event = wait_event
        else:
            self.wait_event = None

    @staticmethod
    def _fn(
        x,
        idevice,
        odevice,
        transfer_stream=None,
        target_stream=None,
        wait_event=None,
    ):
        if x.device != idevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {idevice}"
            )
        if transfer_stream is not None and target_stream is not None:
            if wait_event is not None:
                if isinstance(wait_event, RepeatedEvent):
                    transfer_stream.wait_event(wait_event.last_event)
                else:
                    transfer_stream.wait_event(wait_event)
            else:
                warn(
                    "Peer-to-peer device transfer with wait_event = None detected. Results may not be accurate."
                )
            # Transfer should be initiated on source device
            with torch.cuda.stream(transfer_stream):
                out = x.to(odevice, non_blocking=True)
            # Don't mess with x's memory until transfer is completed
            x.record_stream(transfer_stream)
            # Target stream should wait until transfer is complete
            target_stream.wait_stream(transfer_stream)
            return out

        if odevice.type == "cuda":
            return x.to(odevice, non_blocking=True)
        return x.to(odevice)

    @staticmethod
    def fn(todevice, x, /):
        return todevice._fn(
            x,
            todevice.ispec.device,
            todevice.ospec.device,
            todevice.ispec.transfer_stream,
            todevice.ospec.compute_stream,
            todevice.wait_event,
        )

    @staticmethod
    def adj_fn(todevice, x, /):
        return todevice._fn(
            x,
            todevice.ospec.device,
            todevice.ispec.device,
            todevice.ospec.transfer_stream,
            todevice.ispec.compute_stream,
            todevice.wait_event,
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
            irepr = f"{self.ispec.device}, compute: 0x{self.ispec.compute_stream:x}, transfer: 0x{self.ispec.transfer_stream:x}"
        else:
            irepr = f"{self.ispec.device}"
        if (
            self.ospec.compute_stream is not None
            or self.ospec.transfer_stream is not None
        ):
            orepr = f"{self.ospec.device}, compute: 0x{self.ospec.compute_stream:x}, transfer: 0x{self.ospec.transfer_stream:x}"
        else:
            orepr = f"{self.ospec.device}"
        if self.wait_event is not None:
            wait_event_repr = f"on:{repr(self.wait_event)},"
        else:
            wait_event_repr = ""
        out = f"({wait_event_repr}{irepr} -> {orepr})"
        out = INDENT.indent(out)
        return out
