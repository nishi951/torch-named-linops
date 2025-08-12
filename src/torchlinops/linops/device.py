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
        wait_on_input: bool = True,
    ):
        super().__init__(NS(ioshape))
        self.idevice = torch.device(idevice)
        self.odevice = torch.device(odevice)

        if self.idevice.type == "cuda" and self.odevice.type == "cuda":
            if istream is None and self.idevice.type == "cuda":
                self.istream = torch.cuda.default_stream(self.idevice)
            else:
                self.istream = None
            if ostream is None and self.odevice.type == "cuda":
                self.ostream = torch.cuda.default_stream(self.odevice)
            else:
                self.ostream = None
        else:
            self.istream = None
            self.ostream = None

    @staticmethod
    def fn(linop, x, /):
        if x.device != linop.idevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {linop.idevice}"
            )
        if linop.istream is not None and linop.ostream is not None:
            linop.ostream.wait_stream(linop.istream)
            with torch.cuda.stream(linop.ostream):
                out = x.to(linop.odevice, non_blocking=True)
            return out
        return x.to(linop.odevice, non_blocking=True)

    @staticmethod
    def adj_fn(linop, x, /):
        if x.device != linop.odevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {linop.odevice}"
            )
        if linop.istream is not None and linop.ostream is not None:
            linop.istream.wait_stream(linop.ostream)
            with torch.cuda.stream(linop.istream):
                out = x.to(linop.idevice, non_blocking=True)
            return out
        return x.to(linop.idevice, non_blocking=True)

    def adjoint(self):
        adj = copy(self)
        adj._shape = adj._shape.H
        adj.idevice, adj.odevice = self.odevice, self.idevice
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
        out = f"({self.idevice} -> {self.odevice})"
        out = INDENT.indent(out)
        return out
