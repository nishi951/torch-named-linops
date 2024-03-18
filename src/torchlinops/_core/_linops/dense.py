from collections import defaultdict

from einops import einsum

from .namedlinop import NamedLinop
from .nameddim import NamedDimension as ND


class Dense(NamedLinop):
    """
    Example:
    x: [A, Nx, Ny]
    weightshape: [A, T]
    oshape: [T, Nx, Ny]

    x: [M, N]
    weightshape: [M, M]
    oshape: [M, N]
    - Internally, shapes are renamed
    - second "M" dimension acts on the input, so the above is equivalent to
    x: [M1, N]
    weightshape: [M0, M1]
    oshape: [M0, N]

    """

    def __init__(self, weight, weightshape, ishape, oshape):
        super().__init__(ishape, oshape)
        self.weight = weight
        self.weightshape = ND.from_tuple(weightshape)
        self.weight_ishape = set(self.weightshape) & set(self.ishape)
        self.ishape_only = set(self.ishape) - set(self.weight_ishape)
        self.weight_oshape = set(self.weightshape) & set(self.oshape)
        self.oshape_only = set(self.oshape) - set(self.weight_oshape)

        self.forward_einstr = f"{self.einstr(self.ishape)},{self.einstr(self.weightshape)}->{self.einstr(self.oshape)}"
        self.adj_einstr = f"{self.einstr(self.oshape)},{self.einstr(self.weightshape)}->{self.einstr(self.ishape)}"

    @staticmethod
    def einstr(arr):
        """
        tup: Iterable of str-able objects
        """
        return " ".join(str(s) for s in arr)

    @staticmethod
    def rename_shapes(ishape, oshape, weightshape):
        repeat_dims = defaultdict(int)
        _ishape = []
        for dim in ishape:
            _ishape.append(dim + 1)
        _weightshape = []
        for dim in weightshape:
            repeat_dims[dim] += 1
            _weightshape.append(dim + repeat_dims[dim])
        _oshape = []
        for dim in oshape:
            _oshape.append(dim + repeat_dims[dim])
        return _ishape, _oshape, _weightshape

    def forward(self, x):
        return self.fn(x, self.weight)

    def fn(self, x, /, weight):
        return einsum(x, weight, self.forward_einstr)

    def adj_fn(self, x, /, weight):
        return einsum(x, weight, self.adj_einstr)

    def normal_fn(self, x, /, weight):
        return self.adj_fn(self.fn(x, weight), weight)

    def adjoint(self):
        return type(self)(
            self.weight.conj(), self.weightshape, self.oshape, self.ishape
        )

    def normal(self, inner=None):
        """
        If no inner, consolidate two Dense's into a single Dense
        ishape: [A B X Y]
        oshape: [C D X Y]
        wshape: [A B C D]

        Needs to become
        ishape: [A B X Y]
        oshape: [A1 B1 X Y]
        wshape: [A B A1 B1]

        New weight is attained as
        einsum(weight.conj(), weight, 'A1 B1 C D, A B C D -> A B A1 B1')
        """
        if inner is None:
            # Convert weight
            weight_conj_shape = []
            new_weightshape = []
            for dim in self.weightshape:
                if dim in self.ishape:
                    # Duplicate
                    weight_conj_shape.append(dim + 1)
                    new_weightshape.append(dim)
                else:
                    # Keep/sum over
                    weight_conj_shape.append(dim)
            new_weightshape = new_weightshape + [d + 1 for d in new_weightshape]
            new_weight_einstr = f"{self.einstr(weight_conj_shape)},{self.einstr(self.weightshape)}->{self.einstr(new_weightshape)}"
            new_weight = einsum(self.weight.conj(), self.weight, new_weight_einstr)

            new_oshape = []
            for dim in self.ishape:
                if dim in self.weightshape:
                    new_oshape.append(dim + 1)
                else:
                    new_oshape.append(dim)
            return type(self)(new_weight, new_weightshape, self.ishape, new_oshape)
        pre = copy(self)
        pre.oshape = inner.ishape
        post = copy(self).H
        post.ishape = inner.oshape
        return post @ inner @ pre

    def split_forward(self, ibatch, obatch):
        weight = self.split_forward_fn(ibatch, obatch, self.weight)
        return type(self)(weight, self.weightshape, self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /, weight):
        weightbatch = [slice(None)] * len(self.weightshape)
        for dim, batch in zip(self.ishape, ibatch):
            if dim in self.weightshape:
                weightbatch[self.weightshape.index(dim)] = batch
        for dim, batch in zip(self.oshape, ibatch):
            if dim in self.weightshape:
                weightbatch[self.weightshape.index(dim)] = batch
        return weight[weightbatch]

    def size(self, dim: str):
        return self.size_fn(dim, self.weight)

    def size_fn(self, dim: str, weight):
        if dim in self.weightshape:
            return weight.shape[self.weightshape.index(dim)]
        return None
