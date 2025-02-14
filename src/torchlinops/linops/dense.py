from typing import Optional
from torch import Tensor
from copy import copy, deepcopy

from einops import einsum
import torch.nn as nn

from .namedlinop import NamedLinop
from .nameddim import ND, NS, NamedShape, Shape

__all__ = ["Dense"]


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
        self,
        weight: Tensor,
        weightshape: Shape,
        ishape: Shape,
        oshape: Shape,
        broadcast_dims: Optional[list] = None,
    ):
        """
        broadcast_dims : list
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
        return self.fn(self, x, self.weight)

    @staticmethod
    def fn(linop, x, /, weight):
        return einsum(x, weight, linop.forward_einstr)

    @staticmethod
    def adj_fn(linop, x, /, weight):
        return einsum(x, weight.conj(), linop.adj_einstr)

    @staticmethod
    def normal_fn(linop, x, /, weight):
        return linop.adj_fn(linop.fn(x, weight), weight)

    def adjoint(self):
        adj = copy(self)
        adj.weight = nn.Parameter(
            self.weight.conj(), requires_grad=adj.weight.requires_grad
        )
        adj._shape = adj._shape.H
        adj._update_suffix(adjoint=self._name is not None)
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
        new_oshape = []
        weight_conj_shape = list(deepcopy(self.weightshape))
        wdiag_shape = []
        wout_shape = []
        win_shape = []
        used_shapes = self.ishape + self.oshape
        shape_updates = {}
        # Make new ishape and weight shape
        # Rules:
        # New weightshape
        #   If dim appears in ishape and weightshape but not oshape -> increment
        #   If dim appears in ishape and weightshape AND oshape -> don't increment
        #   If dim doesn't appear in ishape or weightshape -> don't add it to new weightshape
        # Other rules:
        # new ishape is same as old ishape
        # new oshape is ishape but updated with new dimensions
        for dim in self.ishape:
            if dim in self.weightshape:
                if dim not in self.oshape:
                    win_shape.append(dim)
                    new_dim = dim.next_unused(used_shapes)
                    shape_updates[dim] = new_dim
                    wout_shape.append(new_dim)
                else:
                    wdiag_shape.append(dim)
                    new_dim = dim
                i = weight_conj_shape.index(dim)
                weight_conj_shape[i] = new_dim
            else:
                new_dim = dim
            new_oshape.append(new_dim)

        if inner is None:
            # Consolidate dense and dense adjoint into single dense
            new_weight_shape = wdiag_shape + wout_shape + win_shape
            einstr = shapes2einstr(
                self.weightshape,
                weight_conj_shape,
                new_weight_shape,
            )
            new_weight = einsum(self.weight, self.weight.conj(), einstr)
            normal = type(self)(
                new_weight,
                tuple(new_weight_shape),
                self.ishape,
                new_oshape,
            )
            normal._name = self._name
            normal._update_suffix(normal=self._name is not None)
            normal._shape_updates = shape_updates
            return normal
        # Note: Only update _shape_updates when there's no inner linop
        # Inner linops always handle shape updates
        # Basically any time dim.next_unused is called is when
        # shape updates need to happen
        normal = super().normal(inner)
        return normal

    def split_forward(self, ibatch, obatch):
        weight = self.split_forward_fn(ibatch, obatch, self.weight)
        out = copy(self)
        out.weight = nn.Parameter(weight, requires_grad=self.weight.requires_grad)
        return out

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
        if dim in self.broadcast_dims:
            return None
        if dim in self.weightshape:
            return weight.shape[self.weightshape.index(dim)]
        return None


def shapes2einstr(shape1, shape2, oshape):
    """Takes 3 tuples and produces the corresponding einsum string

    Examples
    --------

    """

    def to_str(shape):
        return " ".join(str(s) for s in shape)

    return f"{to_str(shape1)},{to_str(shape2)}->{to_str(oshape)}"
