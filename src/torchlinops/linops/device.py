from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional
from warnings import warn
import logging

import torch
from torch.cuda import Event, Stream, default_stream

import torchlinops.config as config
from torchlinops.utils import INDENT, RepeatedEvent, default_to

from ..nameddim import NamedShape as NS, Shape
from .identity import Identity
from .namedlinop import NamedLinop, ForwardedAttribute

__all__ = ["ToDevice", "DeviceSpec", "clear_transfer_streams_registry"]

logger = logging.getLogger("torchlinops")


def _log_transfer(msg):
    if config.log_device_transfers:
        logger.info(msg)


@dataclass
class DeviceSpec:
    """Lightweight data structure for holding useful CUDA-related objects for multi-GPU computation.

    Attributes
    ----------
    device : torch.device
        The device for computation and transfers.
    compute_stream : Stream, optional
        Stream used for computation on this device. Set automatically by ``p2p_setup``.
    transfer_stream : Stream, optional
        Stream used for data transfers to/from this device. Obtained from a registry
        to enable stream reuse across transfers.

    Methods
    -------
    p2p_setup(other_device)
        Configure compute and transfer streams for peer-to-peer transfers.
    get_transfer_stream(source_device, target_device)
        Get or create a transfer stream for a source/target device pair.
    """

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
        """Sets up compute and transfer streams for peer2peer transfers, if not set yet.

        Parameters
        ----------
        other_device : torch.device
            The other device involved in the peer-to-peer transfer.
        """
        if (
            self.device.type == "cuda" and other_device.type == "cuda"
        ):  # pragma: no cover
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

        Streams are cached in a registry to enable reuse. Each source/target device
        pair gets a dedicated transfer stream.

        Parameters
        ----------
        source_device : torch.device
            The source device for transfers.
        target_device : torch.device
            The target device for transfers.

        Returns
        -------
        Stream
            A CUDA stream for performing transfers.
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
        Source (input) device specification containing device and stream info.
    ospec : DeviceSpec
        Target (output) device specification containing device and stream info.
    is_gpu2gpu : bool
        True if both source and target devices are CUDA devices.
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
        idevice : DeviceSpec | torch.device | None
            Source (input) device specification.
        odevice : DeviceSpec | torch.device | None
            Target (output) device specification.
        ioshape : Shape, optional
            Named dimensions (same for input and output since this is diagonal).
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

        if (
            self.ispec.device.type == "cuda" and self.ospec.device.type == "cuda"
        ):  # pragma: no cover
            self.is_gpu2gpu = True
        else:
            self.is_gpu2gpu = False

    @staticmethod
    def _fn(
        x,
        ispec: DeviceSpec,
        ospec: DeviceSpec,
        input_listener: Optional[Event] = None,
    ):
        idevice, odevice = ispec.device, ospec.device
        if x.device != idevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {idevice}"
            )

        # GPU -> GPU
        if idevice.type == "cuda" and odevice.type == "cuda":  # pragma: no cover
            if input_listener is None:
                warn(
                    "Peer-to-peer device transfer with input_listener = None detected. Results may not be accurate."
                )
            return _gpu2gpu_transfer(
                x,
                odevice,
                ispec.transfer_stream,
                ospec.compute_stream,
                input_listener,
            )
        elif idevice.type == "cuda" and odevice.type == "cpu":  # pragma: no cover
            # GPU -> CPU requires additional synchronization, see:
            # https://github.com/pytorch/pytorch/issues/127612
            return x.to(odevice, non_blocking=False)

        # CPU -> GPU or CPU -> CPU
        return x.to(odevice, non_blocking=True)

    @staticmethod
    def fn(todevice, x, /):
        return todevice._fn(
            x,
            todevice.ispec,
            todevice.ospec,
            todevice.input_listener,
        )

    @staticmethod
    def adj_fn(todevice, x, /):
        return todevice._fn(
            x,
            todevice.ospec,
            todevice.ispec,
            todevice.input_listener,
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
        if self.input_listener is not None and self.is_gpu2gpu:
            input_listener_repr = f"on: {self.input_listener.event_id:x}"
        else:
            input_listener_repr = ""
        out = f"({input_listener_repr} | {irepr} -> {orepr})"
        out = INDENT.indent(out)
        return out


def _gpu2gpu_transfer(x, odevice, transfer_stream, target_stream, input_listener):
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
    input_listener : Event
        The CUDA event that the transfer stream should wait for before initiating transfer.
        Allows fine-grained control of transfer timing. The event should be queued to
        record() when x is ready to be transferred.

    Returns
    -------
    Tensor
        The tensor, on the target device.
    """
    with torch.cuda.stream(transfer_stream):
        if input_listener is not None:
            _log_transfer(f"Stream {transfer_stream} waiting on event {input_listener}")
            transfer_stream.wait_event(input_listener)
        _log_transfer(
            f"Transferring tensor from {x.device} to {odevice}, "
            f"shape={x.shape}, size_bytes={x.element_size() * x.nelement()}"
        )
        out = x.to(odevice, non_blocking=True)
    # Don't mess with x's memory until transfer is completed
    x.record_stream(transfer_stream)
    # Target stream should wait until transfer is complete
    _log_transfer(
        f"Target stream cuda:{target_stream.device_index}:{target_stream} waiting for transfer stream {transfer_stream}"
    )
    target_stream.wait_stream(transfer_stream)
    return out


# Registry to keep track of transfer streams already created.
_TRANSFER_STREAMS_REGISTRY = {}


def clear_transfer_streams_registry() -> None:
    """Clear the transfer streams registry.

    This is useful for testing to ensure a clean state between tests.
    The registry caches CUDA streams to enable reuse across transfers.
    """
    _TRANSFER_STREAMS_REGISTRY.clear()
