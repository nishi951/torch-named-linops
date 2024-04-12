import torch

from .nameddim import NS
from .namedlinop import NamedLinop

__all__ = ["Add"]


class Add(NamedLinop):
    def __init__(self, *linops, strict=True):
        if strict:
            assert all(
                linop.ishape == linops[0].ishape for linop in linops
            ), "All linops must share same ishape"
            assert all(
                linop.oshape == linops[0].oshape for linop in linops
            ), "All linops must share same oshape"
        super().__init__(NS(linops[0].ishape, linops[0].oshape))
        self.linops = linops

    def forward(self, x):
        return sum(linop(x) for linop in self.linops)

    def adjoint(self, x):
        return sum(linop.H(x) for linop in self.linops)

    def fn(self, x: torch.Tensor, /, data_list):
        assert (
            len(self.linops) == len(data_list)
        ), f"Length {len(data_list)} data_list does not match length {len(self.linops)} chain linop"
        return sum(linop.fn(x, *data) for linop, data in zip(self.linops, data_list))

    def adj_fn(self, x: torch.Tensor, /, data_list):
        assert (
            len(self.linops) == len(data_list)
        ), f"Length {len(data_list)} data_list does not match length {len(self.linops)} chain adjoint linop"
        return sum(
            linop.adj_fn(x, *data) for linop, data in zip(self.linops, data_list)
        )

    def normal_fn(self, x: torch.Tensor, /, data_list):
        # Note: Alternatively, make every possible combination of terms? Might be faster in some cases?
        return self.adj_fn(self.fn(x, data_list), data_list)

    def split_forward(self, ibatches, obatches):
        """ibatches, obatches specified according to the shape of the
        forward op
        """
        linops = [
            linop.split(ibatch, obatch)
            for linop, ibatch, obatch in zip(self.linops, ibatches, obatches)
        ]
        return type(self)(*linops)

    def split_forward_fn(self, ibatches, obatches, data_list):
        """Split data into batches
        ibatches, obatches specified according to the shape of the
        forward op
        """
        data = [
            linop.split_forward_fn(ibatch, obatch, *data)
            for linop, ibatch, obatch, data in zip(
                self.linops, ibatches, obatches, data_list
            )
        ]
        return data

    def size(self, dim):
        for linop in self.linops:
            out = linop.size(dim)
            if out is not None:
                return out

    def size_fn(self, dim, data):
        for linop, data in zip(self.linops, data):
            out = linop.size_fn(dim, data)
            if out is not None:
                return out
        return None

    @property
    def dims(self):
        return set().union(*[linop.dims for linop in self.linops])

    @property
    def H(self):
        """Adjoint operator"""
        if self._adj is None:
            linops = list(linop.H for linop in reversed(self.linops))
            _adj = type(self)(*linops)
            self._adj = [_adj]  # Prevent registration as a submodule
        return self._adj[0]

    @property
    def N(self):
        """Normal operator (is this really necessary?)"""
        if self._normal is None:
            linops = list(linop.H for linop in reversed(self.linops)) + list(
                self.linops
            )
            _normal = type(self)(*linops)
            self._normal = [_normal]  # Prevent registration as a submodule
        return self._normal[0]

    def split(self, *iobatches):
        """For compatibility with NamedLinop"""
        ibatches = iobatches[: len(iobatches) // 2]
        obatches = iobatches[len(iobatches) // 2 :]
        return self.split_forward(ibatches, obatches)

    def adj_split(self, *iobatches):
        ibatches = iobatches[: len(iobatches) // 2]
        obatches = iobatches[len(iobatches) // 2 :]
        return self.split_forward(obatches, ibatches).H

    def split_fn(self, *iobatchesdata):
        """Return split versions of the data that can be passed
        into fn and adj_fn to produce split versions
        """
        ibatches = iobatchesdata[: len(iobatchesdata) // 3]
        obatches = iobatchesdata[len(iobatchesdata) // 3 : len(iobatchesdata) * 2 // 3]
        data = iobatchesdata[len(iobatchesdata) * 2 // 3 :]
        return self.split_forward_fn(ibatches, obatches, data)

    def adj_split_fn(self, *iobatchesdata):
        ibatches = iobatchesdata[: len(iobatchesdata) // 3]
        obatches = iobatchesdata[len(iobatchesdata) // 3 : len(iobatchesdata) * 2 // 3]
        data = iobatchesdata[len(iobatchesdata) * 2 // 3]
        return self.split_forward_fn(obatches, ibatches, data)

    def flatten(self):
        return list(self.linops)

    def __getitem__(self, idx):
        return self.linops[idx]

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        linop_chain = "\n\t".join(repr(linop) for linop in self.linops)
        return f"{self.__class__.__name__}(\n\t{linop_chain}\n)"
