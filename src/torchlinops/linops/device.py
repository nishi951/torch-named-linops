from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional
import logging

import torch
from torch.cuda import Stream, default_stream, Event, current_stream

import torchlinops.config as config
from torchlinops.cuda_trace import cuda_logger
from torchlinops.utils import INDENT, default_to

from ..nameddim import NamedShape as NS, Shape
from .identity import Identity
from .namedlinop import NamedLinop

__all__ = ["ToDevice", "DeviceSpec"]

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
        Stream used for transfers from this device to another GPU. Set automatically by ``p2p_setup``.

    Methods
    -------
    p2p_setup(other_device)
        Configure compute and transfer streams for peer-to-peer transfers.
    """

    device: Any = field(default_factory=lambda: torch.device("cpu"))
    """Device for the streams."""
    compute_stream: Optional[Stream] = None
    """Stream used for computation."""
    transfer_stream: Optional[Stream] = None
    """Stream used for transfers from this device."""

    def __post_init__(self):
        """Ensure self.device is a proper torch.device."""
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    def p2p_setup(self, other_device):
        """Sets up compute stream for peer2peer transfers, if not set yet.

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
                self.transfer_stream = Stream(self.device)

    @property
    def type(self):
        """Passthrough for torch.device.type."""
        return self.device.type


class ToDevice(NamedLinop):
    """Transfer tensors between devices as a named linear operator.

    The forward operation moves a tensor from ``idevice`` to ``odevice``.
    The adjoint reverses the direction. The normal $T^H T$ is the identity
    (device round-trip is lossless).

    For CUDA-to-CUDA transfers, dedicated streams are used for asynchronous
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
        wait_for_event: Optional[Event] = None,
    ):
        """Transfer a tensor between devices.

        Handles CPU↔GPU and GPU↔GPU transfers. For GPU→GPU, uses dedicated
        transfer and compute streams for asynchronous pipelined execution.

        Parameters
        ----------
        x : Tensor
            The tensor to transfer. Must be on ``ispec.device``.
        ispec : DeviceSpec
            Source device specification.
        ospec : DeviceSpec
            Target device specification.
        wait_for_event : Event, optional
            CUDA event to wait on before starting the transfer.

        Returns
        -------
        Tensor
            The tensor on ``ospec.device``.

        Notes
        -----
        Copying from CPU → GPU or GPU → CPU may result in strange behavior:
            https://github.com/pytorch/pytorch/issues/127612
        """
        idevice, odevice = ispec.device, ospec.device
        if x.device != idevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {idevice}"
            )

        # GPU -> GPU
        if idevice.type == "cuda" and odevice.type == "cuda":  # pragma: no cover
            return _gpu2gpu_transfer(
                x,
                ospec.compute_stream,
                ispec.transfer_stream,
                wait_for_event,
            )
        elif idevice.type == "cuda" and odevice.type == "cpu":  # pragma: no cover
            # GPU -> CPU requires non_blocking=False for stability.
            # This is usually ok since GPU -> CPU usually happens at the "end" of a computation,
            # when synchronization isn't a problem.
            return x.to(odevice, non_blocking=False)

        # CPU -> GPU or CPU -> CPU
        # Need to be careful not to overwrite x in-place too quickly after calling this.
        # However, we riskily choose non_blocking=True because CPU -> GPU typically occurs
        # at the "beginning" of a parallelized computation, where non-blocking makes a big difference.
        return x.to(odevice, non_blocking=True)

    @staticmethod
    def fn(todevice, x, context):
        return todevice._fn(
            x,
            todevice.ispec,
            todevice.ospec,
            wait_for_event=context.start_event,
        )

    @staticmethod
    def adj_fn(todevice, x, context):
        return todevice._fn(
            x,
            todevice.ospec,
            todevice.ispec,
            wait_for_event=context.start_event,
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

    @staticmethod
    def split(linop, tile):
        """Return a new instance"""
        return copy(linop)

    def __repr__(self):
        """Helps prevent recursion error caused by .H and .N"""
        if self.ispec.transfer_stream is not None:  # pragma: no cover
            irepr = f"{self.ispec.device}, transfer: 0x{self.ispec.transfer_stream.cuda_stream:x}"
        else:
            irepr = f"{self.ispec.device}"
        if self.ospec.compute_stream is not None:  # pragma: no cover
            orepr = f"{self.ospec.device}, compute: 0x{self.ospec.compute_stream.cuda_stream:x}"
        else:
            orepr = f"{self.ospec.device}"
        out = f"({irepr} -> {orepr})"
        out = INDENT.indent(out)
        return out


def _gpu2gpu_transfer(
    x,
    target_stream,
    transfer_stream=None,
    wait_for_event: Optional[Event] = None,
):  # pragma: no cover
    """Perform efficient gpu-gpu transfer with a dedicated transfer stream.

    Parameters
    ----------
    x : Tensor
        The torch tensor to be transferred.
    target_stream : Stream
        The stream on the target device that needs `x` for computation.
    transfer_stream : Stream
        The stream on the source device on which to queue the transfer.
    wait_for_event : Event, optional
        Optional event for transfer stream to wait on before proceeding.

    Returns
    -------
    Tensor
        The tensor, on the target device.
    """

    if x.device != target_stream.device:
        odevice = target_stream.device
        if transfer_stream is None:
            raise ValueError(f"Multi-GPU transfer requires transfer_stream != None")

        if config.log_cuda_events:
            src_id = cuda_logger.implicit_node(f"default_stream:{x.device}", x.device)
            cuda_logger.wait(
                f"ToDevice:{x.device}\u2192{odevice}:transfer",
                x.device,
                [src_id],
                reason="wait_stream",
            )

        if wait_for_event is not None:
            # Used when this is run as the first step in a Chain
            logger.debug(f"Waiting on parent event: {wait_for_event}")
            transfer_stream.wait_event(wait_for_event)
        else:
            logger.debug(f"Waiting on current device's stream: {x.device}")
            # Used when this is run as a later step in a Chain
            # Sometimes this is not what we want because there might be a lot of work on the default stream
            # that we want to parallelize with this transfer.
            transfer_stream.wait_stream(current_stream(x.device))

        with torch.cuda.stream(transfer_stream):
            _log_transfer(
                f"Transferring tensor from {x.device} to {odevice}, "
                f"shape={x.shape}, size_bytes={x.element_size() * x.nelement()}"
            )
            out = x.to(odevice, non_blocking=True)
            # Don't mess with x's memory until transfer is completed
            x.record_stream(transfer_stream)

        if config.log_cuda_events:
            transfer_id = cuda_logger.record(
                f"ToDevice:{x.device}\u2192{odevice}:transfer-done", x.device
            )
            cuda_logger.wait(
                f"ToDevice:{x.device}\u2192{odevice}:target-wait",
                odevice,
                [transfer_id],
                reason="wait_stream",
            )

        # Target stream should wait until transfer is complete
        _log_transfer(
            f"Target stream cuda:{target_stream.device_index}:{target_stream} waiting for transfer stream {transfer_stream}"
        )
        target_stream.wait_stream(transfer_stream)
    else:
        # Same device - do nothing
        out = x
    return out
