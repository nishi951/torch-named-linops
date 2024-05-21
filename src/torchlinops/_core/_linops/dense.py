from copy import copy
from typing import Optional, List

from einops import einsum
import torch.nn as nn

from .namedlinop import NamedLinop
from .nameddim import ND, NS, NamedShape


class Dense(NamedLinop):
    """
    Example:
    x: [A, Nx, Ny]
    weightshape: [A, T]
    oshape: [T, Nx, Ny]

    shape:
        diag.ioshape = [Nx, Ny]
        dense.ishape = [A]
        dense.oshape = [T]


    # Diagonal/elementwise dimensions
    # Einsum can infer which dims are shared (literally the point of
    # the summation notation)
    x: [C, A, Nx, Ny]
    weight: [C, A, A1]
    oshape: [C, A1, Nx, Ny]
    """

    def __init__(
        self, weight, weightshape, ishape, oshape, broadcast_dims: Optional[List] = None
    ):
        """
        broadcast_dims : List
            A list of the dimensions of weight that are intended to be broadcasted over the input.
            As such, they are excluded from splitting.
        """
        super().__init__(NS(ishape, oshape))
        self.weight = nn.Parameter(weight, requires_grad=False)
        self._shape.add("weightshape", weightshape)

        broadcast_dims = broadcast_dims if broadcast_dims is not None else []
        self._shape.add("broadcast_dims", broadcast_dims)

    @property
    def weightshape(self):
        return self._shape.lookup("weightshape")

    @property
    def broadcast_dims(self):
        return self._shape.lookup("broadcast_dims")

    @property
    def forward_einstr(self):
        return f"{self.einstr(self.ishape)},{self.einstr(self.weightshape)}->{self.einstr(self.oshape)}"

    @property
    def adj_einstr(self):
        return f"{self.einstr(self.oshape)},{self.einstr(self.weightshape)}->{self.einstr(self.ishape)}"

    @staticmethod
    def einstr(arr):
        """
        tup: Iterable of str-able objects
        """
        return " ".join(str(s) for s in arr)

    def forward(self, x):
        return self.fn(x, self.weight)

    def fn(self, x, /, weight):
        return einsum(x, weight, self.forward_einstr)

    def adj_fn(self, x, /, weight):
        return einsum(x, weight, self.adj_einstr)

    def normal_fn(self, x, /, weight):
        return self.adj_fn(self.fn(x, weight), weight)

    def adjoint(self):
        adj = type(self)(
            self.weight.conj(),
            self.weightshape,
            self._shape.H.ishape,
            self._shape.H.oshape,
            self.broadcast_dims,
        )
        return adj

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

        -----
        ishape: [C A]
        oshape: [C1 A]
        wshape = [C C1]

        Needs to become
        ishape: [C A]
        oshape: [C2 A]
        wshape = [C C2]

        einsum(weight.conj(), weight, 'C1 C2, C C1 -> C C2)


        """
        if inner is None:
            # Convert weight
            weight_conj_shape = []
            new_weightshape = []
            for dim in self.weightshape:
                if dim in self.ishape:
                    # Dense-like
                    # Get the new name of the output dim
                    new_dim = dim.next_unused(self.ishape + self.oshape)
                    weight_conj_shape.append(new_dim)
                    new_weightshape.extend([dim, new_dim])
                else:
                    # Keep/sum over the output dims of the weight
                    weight_conj_shape.append(dim)
            new_weight_einstr = f"{self.einstr(weight_conj_shape)},{self.einstr(self.weightshape)}->{self.einstr(new_weightshape)}"
            new_weight = einsum(self.weight.conj(), self.weight, new_weight_einstr)

            new_oshape = []
            for dim in self.ishape:
                if dim in self.weightshape:
                    # Re-create the new output dim here
                    # But now in the order of the ishape
                    new_oshape.append(dim.next_unused(self.ishape + self.oshape))
                else:
                    new_oshape.append(dim)
            normal = type(self)(new_weight, new_weightshape, self.ishape, new_oshape)
            return normal
        return super().normal(inner)

    def split_forward(self, ibatch, obatch):
        weight = self.split_forward_fn(ibatch, obatch, self.weight)
        return type(self)(weight, self.weightshape, self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /, weight):
        weightbatch = [slice(None)] * len(self.weightshape)
        for dim, batch in zip(self.ishape, ibatch):
            if dim in self.weightshape and dim not in self.broadcast_dims:
                weightbatch[self.weightshape.index(dim)] = batch
        for dim, batch in zip(self.oshape, obatch):
            if dim in self.weightshape and dim not in self.broadcast_dims:
                weightbatch[self.weightshape.index(dim)] = batch
        return weight[weightbatch]

    def size(self, dim: str):
        return self.size_fn(dim, self.weight)

    def size_fn(self, dim: str, weight):
        if dim in self.weightshape:
            return weight.shape[self.weightshape.index(dim)]
        return None
