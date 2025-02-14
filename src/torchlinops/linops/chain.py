from typing import Optional
import sys
import traceback

import torch
import torch.nn as nn

from .namedlinop import NamedLinop
from .nameddim import NS, isequal

from torchlinops.utils import INDENT


class Chain(NamedLinop):
    """A sequence or composition of linops"""

    def __init__(self, *linops, name: Optional[str] = None):
        """
        Parameters
        ----------
        *linops : list
            Linops in order of execution
            i.e. if `linops = [A, B, C]`, then mathematically, the linop in question is `CBA`

        """
        super().__init__(NS(linops[0].ishape, linops[-1].oshape), name=name)
        self.linops = nn.ModuleList(list(linops))
        self._check_inputs_outputs()

    def _check_inputs_outputs(self):
        curr_shape = self.ishape
        for i, linop in enumerate(self.linops):
            if not isequal(linop.ishape, curr_shape):
                raise ValueError(
                    f"Mismatched shape: expected {linop.ishape}, got {curr_shape} at input to {linop}. Full stack: {self}, index {i}"
                )
            curr_shape = linop.oshape

    def forward(self, x):
        for linop in self.linops:
            x = linop(x)
        return x

    @staticmethod
    def fn(chain, x: torch.Tensor, /, data_list):
        assert len(chain.linops) == len(data_list), (
            f"Length {len(data_list)} data_list does not match length {len(chain.linops)} chain linop"
        )
        for linop, data in zip(chain.linops, data_list):
            x = linop.fn(x, *data)
        return x

    @staticmethod
    def adj_fn(chain, x: torch.Tensor, /, data_list):
        assert len(chain.linops) == len(data_list), (
            f"Length {len(data_list)} data_list does not match length {len(chain.linops)} chain adjoint linop"
        )
        for linop, data in zip(reversed(chain.linops), reversed(data_list)):
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
        return type(self)(*linops, name=self._name)

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
        out = None
        for linop in self.linops:
            tmp = linop.size(dim)
            if tmp is not None:
                if out is None:
                    out = tmp
                elif out != tmp:
                    raise ValueError(
                        f"Conflicting linop sizes found: {out} and {tmp} for dim {dim} in linop {linop} out of all linops {self.linops}"
                    )
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

    def adjoint(self):
        linops = list(linop.adjoint() for linop in reversed(self.linops))
        return type(self)(*linops, name=self._name)

    def normal(self, inner=None):
        for linop in reversed(self.linops):
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

    @property
    def shape(self):
        return NS(self.linops[0].ishape, self.linops[-1].oshape)

    @shape.setter
    def shape(self, val):
        self.ishape = val.ishape
        self.oshape = val.oshape

    @property
    def ishape(self):
        return self.linops[0].ishape

    @ishape.setter
    def ishape(self, val):
        self.linops[0].ishape = val

    @property
    def oshape(self):
        return self.linops[-1].oshape

    @oshape.setter
    def oshape(self, val):
        self.linops[-1].oshape = val

    def flatten(self):
        return list(self.linops)

    def __getitem__(self, idx):
        linops = self.linops[idx]
        if isinstance(linops, NamedLinop):
            return linops
        return type(self)(*linops, name=self._name)

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        output = ""
        output += INDENT.indent(self.repr_name + "(\n")
        with INDENT:
            for linop in self.linops:
                output += repr(linop) + "\n"
        output += INDENT.indent(")")
        return output
