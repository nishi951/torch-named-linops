from .namedlinop import NamedLinop
from .nameddim import NS

__all__ = ["Identity", "Zero"]


class Identity(NamedLinop):
    def __init__(self, ishape, oshape=None):
        super().__init__(NS(ishape, oshape))

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x, /):
        return x

    @staticmethod
    def adj_fn(linop, x, /):
        return x

    @staticmethod
    def normal_fn(linop, x, /):
        return x

    def split_forward(self, ibatch, obatch):
        # TODO: Allow non-diagonal splitting
        assert ibatch == obatch, "Identity linop must be split identically"
        return self

    def split_forward_fn(self, ibatch, obatch, /):
        assert ibatch == obatch, "Identity linop must be split identically"
        return None

    def size(self, dim: str):
        return self.size_fn(dim)

    def size_fn(self, dim: str):
        return None


class Zero(NamedLinop):
    """Simple linop that always outputs 0, but with the same shape as the input"""

    def __init__(self, ishape, oshape=None):
        super().__init__(NS(ishape, oshape))

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(self, x, /):
        return x.zero_()

    @staticmethod
    def adj_fn(self, x, /):
        return x.zero_()

    @staticmethod
    def normal_fn(self, x, /):
        return x.zero_()

    def split_forward(self, ibatch, obatch):
        return self

    def split_forward_fn(self, ibatch, obatch, /):
        return None

    def size(self, dim: str):
        return self.size_fn(dim)

    def size_fn(self, dim: str):
        return None
