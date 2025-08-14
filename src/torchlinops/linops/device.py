from copy import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor
from torch.cuda import Stream

from torchlinops.utils import INDENT
from .identity import Identity
from .nameddim import ELLIPSES, NS, Shape
from .namedlinop import NamedLinop

__all__ = ["ToDevice"]


class ToDevice(NamedLinop):
    def __init__(
        self,
        idevice: torch.device | str,
        odevice: torch.device | str,
        ioshape: Optional[Shape] = None,
        istream: Optional[Stream] = None,
        ostream: Optional[Stream] = None,
    ):
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
        else:
            self.istream = None
            self.ostream = None

    @staticmethod
    def _fn(x, idevice, odevice, istream=None, ostream=None):
        if x.device != idevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {idevice}"
            )
        if istream is not None and ostream is not None:
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
            x, todevice.idevice, todevice.odevice, todevice.istream, todevice.ostream
        )

    @staticmethod
    def adj_fn(todevice, x, /):
        return todevice._fn(
            x, todevice.odevice, todevice.idevice, todevice.ostream, todevice.istream
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
        out = f"({irepr} -> {orepr})"
        out = INDENT.indent(out)
        return out


def get_stream_device(stream: torch.cuda.Stream) -> torch.device:
    with torch.cuda.stream(stream):
        return torch.device(f"cuda:{torch.cuda.current_device()}")
