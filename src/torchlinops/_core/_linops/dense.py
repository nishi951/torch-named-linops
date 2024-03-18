from collections import defaultdict

from einops import einsum

from .namedlinop import NamedLinop


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
        self.weightshape = weightshape
        self._ishape, self._oshape, self._weightshape = self.__rename_shapes(
            self.ishape, self.oshape, self.weightshape
        )
        self.einstr = f'{" ".join(self._ishape)},{" ".join(self._weightshape)}->{" ".join(self._oshape)}'
        self.adj_einstr = f'{" ".join(self._oshape)},{" ".join(self._weightshape)}->{" ".join(self._ishape)}'

    @staticmethod
    def __rename_shapes(ishape, oshape, weightshape):
        dim_idx = defaultdict(int)
        _ishape = []
        for dim in ishape:
            _ishape.append(dim + str(dim_idx[dim]))
        _weightshape = []
        for dim in weightshape:
            _weightshape.append(dim + str(dim_idx[dim]))
            dim_idx[dim] += 1
        _oshape = []
        for dim in oshape:
            _oshape.append(dim + str(dim_idx[dim]))
        return _ishape, _oshape, _weightshape

    def forward(self, x):
        return self.fn(x, self.weight)

    def fn(self, x, /, weight):
        return einsum(x, weight, self.einstr)

    def adj_fn(self, x, /, weight):
        return einsum(x, weight, self.adj_einstr)

    def normal_fn(self, x, /, weight):
        return self.adj_fn(self.fn(x, weight), weight)

    def get_normal(self, inner=None):
        # Get output shapes
        weightoshape = []
        for dim in self.weightshape:
            if dim in self.oshape:
                weightoshape.append(dim)
        if inner is None:
            # Consolidate back-to-back dense ops into a single op
            normal_einstr = f'{" ".join(self.weightshape)},{" ".join(self.weightshape)} -> {" ".join(weightoshape)},{" ".join(weightoshape)}'
            weight = einsum(self.weight.conj(), self.weight, normal_einstr)
            return type(self)(
                weight, weightoshape + weightoshape, self.ishape, self.ishape
            )
        return self.H @ inner @ self

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
