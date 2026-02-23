from copy import copy
from typing import Optional

import torch
from torch.cuda import Event, Stream

from torchlinops.utils import INDENT, RepeatedEvent

from ..nameddim import NamedShape as NS, Shape
from .identity import Identity
from .namedlinop import NamedLinop

__all__ = ["ToDevice"]


class ToDevice(NamedLinop):
    """Transfer tensors between devices as a named linear operator.

    The forward operation moves a tensor from ``idevice`` to ``odevice``.
    The adjoint reverses the direction. The normal $T^H T$ is the identity
    (device round-trip is lossless).

    For CUDA-to-CUDA transfers, streams and events are used for asynchronous
    pipelined execution.

    Attributes
    ----------
    idevice : torch.device
        Source (input) device.
    odevice : torch.device
        Target (output) device.
    istream : Stream or None
        CUDA stream on the source device for initiating the transfer.
    ostream : Stream or None
        CUDA stream on the target device that waits for the transfer.
    wait_event : Event or None
        Optional event to wait on before starting the transfer.
    """

    def __init__(
        self,
        idevice: torch.device | str,
        odevice: torch.device | str,
        ioshape: Optional[Shape] = None,
        istream: Optional[Stream] = None,
        ostream: Optional[Stream] = None,
        wait_event: Optional[Event] = None,
    ):
        """
        Parameters
        ----------
        idevice : torch.device or str
            Source device.
        odevice : torch.device or str
            Target device.
        ioshape : Shape, optional
            Named dimensions (same for input and output since this is diagonal).
        istream : Stream, optional
            CUDA stream on the source device. Defaults to the device's default stream.
        ostream : Stream, optional
            CUDA stream on the target device. Defaults to the device's default stream.
        wait_event : Event, optional
            A CUDA event to wait on before initiating the transfer.
        """
        super().__init__(NS(ioshape))
        self.idevice = torch.device(idevice)
        self.odevice = torch.device(odevice)

        if self.idevice.type == "cuda" and self.odevice.type == "cuda":
            if istream is None:
                self.istream = torch.cuda.default_stream(self.idevice)
            else:
                if self.idevice != istream.device:
                    raise ValueError(
                        f"stream {istream} must be on {self.idevice} but got {istream.device}"
                    )

                self.istream = istream
            if ostream is None:
                self.ostream = torch.cuda.default_stream(self.odevice)
            else:
                if self.odevice != ostream.device:
                    raise ValueError(
                        f"stream {ostream} must be on {self.odevice} but got {ostream.device}"
                    )
                self.ostream = ostream
            self.wait_event = wait_event
        else:
            self.istream = None
            self.ostream = None
            self.wait_event = None

    @staticmethod
    def _fn(x, idevice, odevice, istream=None, ostream=None, wait_event=None):
        if x.device != idevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {idevice}"
            )
        if istream is not None and ostream is not None:
            if wait_event is not None:
                if isinstance(wait_event, RepeatedEvent):
                    istream.wait_event(wait_event.last_event)
                else:
                    istream.wait_event(wait_event)
            # Transfer should be initiated on source device
            with torch.cuda.stream(istream):
                out = x.to(odevice, non_blocking=True)
            # Don't mess with x's memory until transfer is completed
            x.record_stream(istream)
            # Target stream should wait until transfer is complete
            ostream.wait_stream(istream)
            return out

        if odevice.type == "cuda":
            return x.to(odevice, non_blocking=True)
        return x.to(odevice)

    @staticmethod
    def fn(todevice, x, /):
        return todevice._fn(
            x,
            todevice.idevice,
            todevice.odevice,
            todevice.istream,
            todevice.ostream,
            todevice.wait_event,
        )

    @staticmethod
    def adj_fn(todevice, x, /):
        return todevice._fn(
            x,
            todevice.odevice,
            todevice.idevice,
            todevice.ostream,
            todevice.istream,
            todevice.wait_event,
        )

    def adjoint(self):
        adj = copy(self)
        adj._shape = adj._shape.H
        adj.idevice, adj.odevice = self.odevice, self.idevice
        adj.istream, adj.ostream = self.ostream, self.istream
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
        if self.istream is not None:
            irepr = f"{self.idevice}, 0x{self.istream.cuda_stream:x}"
        else:
            irepr = f"{self.idevice}"
        if self.ostream is not None:
            orepr = f"{self.odevice}, 0x{self.ostream.cuda_stream:x}"
        else:
            orepr = f"{self.odevice}"
        if self.wait_event is not None:
            wait_event_repr = f"on:{repr(self.wait_event)},"
        else:
            wait_event_repr = ""
        out = f"({wait_event_repr}{irepr} -> {orepr})"
        out = INDENT.indent(out)
        return out


def get_stream_device(stream: torch.cuda.Stream) -> torch.device:
    """Return the ``torch.device`` associated with a CUDA stream."""
    with torch.cuda.stream(stream):
        return torch.device(f"cuda:{torch.cuda.current_device()}")
