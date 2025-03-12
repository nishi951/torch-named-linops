from typing import Optional
from copy import copy
import torch
from .namedlinop import NamedLinop
from .identity import Identity
from .nameddim import ELLIPSES, NS, Shape

__all__ = ["ToDevice"]


class ToDevice(NamedLinop):
    def __init__(
        self,
        idevice: torch.device | str,
        odevice: torch.device | str,
        ioshape: Optional[Shape] = None,
    ):
        super().__init__(NS(ioshape))
        self.idevice = torch.device(idevice)
        self.odevice = torch.device(odevice)

    @staticmethod
    def fn(linop, x, /):
        if x.device != linop.idevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {linop.idevice}"
            )
        return x.to(linop.odevice)

    @staticmethod
    def adj_fn(linop, x, /):
        if x.device != linop.odevice:
            raise RuntimeError(
                f"Got input to ToDevice on {x.device} but expected {linop.odevice}"
            )
        return x.to(linop.idevice)

    def normal(self, inner=None):
        if inner is None:
            return Identity()
        return super().normal(inner)

    def split_forward(self, ibatch, obatch):
        """Return a new instance"""
        return copy(self)
