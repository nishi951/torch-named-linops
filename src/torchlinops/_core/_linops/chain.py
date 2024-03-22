import torch
import torch.nn as nn

from .namedlinop import NamedLinop


class Chain(NamedLinop):
    def __init__(self, *linops):
        super().__init__(linops[-1].ishape, linops[0].oshape)
        self.linops = nn.ModuleList(list(linops))
        # self.signatures = [signature(linop.fn) for linop in self.linops]
        # self._check_signatures()
        self._check_inputs_outputs()

    # def _check_signatures(self):
    #     seen = set()
    #     for sig in self.signatures:
    #         for param in sig.parameters.values():
    #             if param.name in seen:
    #                 logger.debug(
    #                     f'{param.name} appears more than once in linop chain.'
    #                 )

    def _check_inputs_outputs(self):
        curr_shape = self.ishape
        for linop in reversed(self.linops):
            if linop.ishape != curr_shape:
                raise ValueError(
                    f"Mismatched shape: expected {linop.ishape}, got {curr_shape} at input to {linop}"
                )
            curr_shape = linop.oshape

    def forward(self, x):
        for linop in reversed(self.linops):
            x = linop(x)
        return x

    def adjoint(self, x):
        for linop in self.linops:
            x = linop(x)
        return x

    def fn(self, x: torch.Tensor, /, data_list):
        assert (
            len(self.linops) == len(data_list)
        ), f"Length {len(data_list)} data_list does not match length {len(self.linops)} chain linop"
        for linop, data in zip(reversed(self.linops), reversed(data_list)):
            x = linop.fn(x, *data)
        return x

    def adj_fn(self, x: torch.Tensor, /, data_list):
        assert (
            len(self.linops) == len(data_list)
        ), f"Length {len(data_list)} data_list does not match length {len(self.linops)} chain adjoint linop"
        for linop, data in zip(self.linops, data_list):
            x = linop.adj_fn(x, data)
        return x

    def normal_fn(self, x: torch.Tensor, /, data_list):
        # fn does the reversing so it's unnecessary to do it here
        # If the normal hasn't been explicitly formed with`.N`, do things the naive way
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
            _normal = None
            for linop in self.linops:
                _normal = linop.normal(inner=_normal)
            # linops = list(linop.H for linop in reversed(self.linops)) + list(
            #     self.linops
            # )
            # _normal = type(self)(*linops)
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
        linops = self.linops[idx]
        if isinstance(linops, NamedLinop):
            return linops
        return type(self)(*linops)

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        linop_chain = "\n\t".join(repr(linop) for linop in self.linops)
        return f"{self.__class__.__name__}(\n\t{linop_chain}\n)"
