from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional
import logging

import torch
from torch.cuda import Stream, default_stream

import torchlinops.config as config
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

    Methods
    -------
    p2p_setup(other_device)
        Configure compute stream for peer-to-peer transfers.
    """

    device: Any = field(default_factory=lambda: torch.device("cpu"))
    """Device for the streams."""
    compute_stream: Optional[Stream] = None
    """Stream used for computation."""

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
            self._transfer_stream = Stream(self.ispec.device)
        else:
            self.is_gpu2gpu = False
            self._transfer_stream = None

    @staticmethod
    def _fn(
        x,
        ispec: DeviceSpec,
        ospec: DeviceSpec,
        transfer_stream=None,
    ):
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
                transfer_stream,
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
            todevice._transfer_stream,
        )

    @staticmethod
    def adj_fn(todevice, x, /):
        return todevice._fn(
            x,
            todevice.ospec,
            todevice.ispec,
            todevice._transfer_stream,
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
        if self._transfer_stream is not None:  # pragma: no cover
            irepr = f"{self.ispec.device}, transfer: 0x{self._transfer_stream.cuda_stream:x}"
        else:
            irepr = f"{self.ispec.device}"
        if self._transfer_stream is not None:  # pragma: no cover
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

    Returns
    -------
    Tensor
        The tensor, on the target device.
    """

    if x.device != target_stream.device:
        odevice = target_stream.device
        if transfer_stream is None:
            raise ValueError(f"Multi-GPU transfer requires transfer_stream != None")
        transfer_stream.wait_stream(default_stream(x.device))
        with torch.cuda.stream(transfer_stream):
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
    else:
        # Same device - do nothing
        out = x
    return out
