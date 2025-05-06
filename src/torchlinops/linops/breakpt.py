from typing import Optional

from .namedlinop import NamedLinop
from .nameddim import NS, Shape

__all__ = ["BreakpointLinop"]


class BreakpointLinop(NamedLinop):
    def __init__(self, ioshape: Optional[Shape] = None):
        super().__init__(NS(ioshape))

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x, /):
        breakpoint()
        return x

    @staticmethod
    def adj_fn(linop, x, /):
        breakpoint()
        return x

    def split_forward(self, ibatch, obatch):
        return self

    def split_forward_fn(self, ibatch, obatch, /, data):
        return None
