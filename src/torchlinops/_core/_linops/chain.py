import sys
import traceback

import torch
import torch.nn as nn

from .namedlinop import NamedLinop
from .nameddim import NS


class Chain(NamedLinop):
    def __init__(self, *linops):
        super().__init__(NS(linops[-1].ishape, linops[0].oshape))
        self.linops = nn.ModuleList(list(linops))
        self._check_inputs_outputs()

    def _check_inputs_outputs(self):
        curr_shape = self.ishape
        for linop in reversed(self.linops):
            if linop.ishape != curr_shape:
                # breakpoint()
                raise ValueError(
                    f"Mismatched shape: expected {linop.ishape}, got {curr_shape} at input to {linop}"
                )
            curr_shape = linop.oshape

    def forward(self, x):
        for linop in reversed(self.linops):
            x = linop(x)
            # print(f'{linop}: {x.shape}')
        return x

    @staticmethod
    def fn(chain, x: torch.Tensor, /, data_list):
        assert (
            len(chain.linops) == len(data_list)
        ), f"Length {len(data_list)} data_list does not match length {len(chain.linops)} chain linop"
        for linop, data in zip(reversed(chain.linops), reversed(data_list)):
            x = linop.fn(x, *data)
        return x

    @staticmethod
    def adj_fn(chain, x: torch.Tensor, /, data_list):
        assert (
            len(chain.linops) == len(data_list)
        ), f"Length {len(data_list)} data_list does not match length {len(chain.linops)} chain adjoint linop"
        for linop, data in zip(chain.linops, data_list):
            x = linop.adj_fn(x, data)
        return x

    @staticmethod
    def normal_fn(chain, x: torch.Tensor, /, data_list):
        # fn does the reversing so it's unnecessary to do it here
        # If the normal hasn't been explicitly formed with`.N`, do things the naive way
        return chain.adj_fn(chain.fn(x, data_list), data_list)

    def split_forward(self, ibatches, obatches):
        """ibatches, obatches specified according to the shape of the
        forward op
        """
        linops = [
            linop.split(linop, ibatch, obatch)
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
        return None

    def size_fn(self, dim, data):
        for linop, data in zip(self.linops, data):
            out = linop.size_fn(dim, data)
            if out is not None:
                return out
        return None

    @property
    def dims(self):
        return set().union(*[linop.dims for linop in self.linops])

    def adjoint(self):
        linops = list(linop.adjoint() for linop in reversed(self.linops))
        return type(self)(*linops)

    def normal(self, inner=None):
        for linop in self.linops:
            inner = linop.normal(inner)
        return inner

    @staticmethod
    def split(chain, *iobatches):
        """For compatibility with NamedLinop"""
        ibatches = iobatches[: len(iobatches) // 2]
        obatches = iobatches[len(iobatches) // 2 :]
        return chain.split_forward(ibatches, obatches)

    @staticmethod
    def adj_split(chain, *iobatches):
        ibatches = iobatches[: len(iobatches) // 2]
        obatches = iobatches[len(iobatches) // 2 :]
        return chain.H.split_forward(obatches, ibatches).H

    @staticmethod
    def split_fn(chain, *iobatchesdata):
        """Return split versions of the data that can be passed
        into fn and adj_fn to produce split versions
        """
        ibatches = iobatchesdata[: len(iobatchesdata) // 3]
        obatches = iobatchesdata[len(iobatchesdata) // 3 : len(iobatchesdata) * 2 // 3]
        data = iobatchesdata[len(iobatchesdata) * 2 // 3 :]
        return chain.split_forward_fn(ibatches, obatches, data)

    @staticmethod
    def adj_split_fn(chain, *iobatchesdata):
        ibatches = iobatchesdata[: len(iobatchesdata) // 3]
        obatches = iobatchesdata[len(iobatchesdata) // 3 : len(iobatchesdata) * 2 // 3]
        data = iobatchesdata[len(iobatchesdata) * 2 // 3]
        return chain.split_forward_fn(obatches, ibatches, data)

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
