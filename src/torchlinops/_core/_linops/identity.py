from .namedlinop import NamedLinop


class Identity(NamedLinop):
    def __init__(self, ioshape):
        super().__init__(ioshape, ioshape)

    def forward(self, x):
        return x

    def fn(self, x, /):
        return x

    def adj_fn(self, x, /):
        return x

    def normal_fn(self, x, /):
        return x

    def split_forward(self, ibatch, obatch):
        return type(self)(self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /):
        assert ibatch == obatch, "Identity linop must be split identically"
        return None

    def size(self, dim: str):
        return self.size_fn(dim)

    def size_fn(self, dim: str):
        return None
