from copy import copy
from inspect import signature

import torch
import torch.nn as nn

__all__ = ['NamedLinop', 'Chain']

class NamedLinop(nn.Module):
    def __init__(self, ishape, oshape):
        """ishape and oshape are symbolic, not numeric
        They also change if the adjoint is taken (!)
        """
        super().__init__()
        self.ishape = ishape
        self.oshape = oshape

        self._adj = None
        self._normal = None

        self._suffix = ''

    # Change the call to self.fn according to the data
    def forward(self, x: torch.Tensor):
        return self.fn(x)

    # Override
    def fn(self, x: torch.Tensor, /, data=None):
        """Placeholder for functional forwa rd operator.
        Non-input arguments should be keyword-only
        self can still be used - kwargs should contain elements
        that may change frequently (e.g. trajectories) and can
        ignore hyperparameters (e.g. normalization modes)
        """
        return x

    # Override
    def adj_fn(self, x: torch.Tensor, /, data=None):
        """Placeholder for functional adjoint operator.
        Non-input arguments should be keyword-only"""
        return x

    # Override
    def normal_fn(self, x: torch.Tensor, /, data=None):
        """Placeholder for efficient functional normal operator"""
        return self.adj_fn(self.fn(x, data), data)

    # Override
    def split_forward(self, ibatch, obatch):
        """Return a new instance"""
        raise NotImplementedError(f'{type(self).__name__} cannot be split.')

    # Override
    def split_forward_fn(self, ibatch, obatch, /, data=None):
        """Return data"""
        raise NotImplementedError(f'{type(self).__name__} cannot be split.')


    # Override
    def size(self, dim: str):
        """Get the size of a particular dim, or return
        None if this linop doesn't determine the size
        """
        return None

    # Override
    def size_fn(self, dim: str, /, data=None):
        """Functional version of size. Determines sizes from kwargs
        kwargs should be the same as the inputs to fn or adj_fn
        Return None if this linop doesn't determine the size of dim
        """
        return None

    # Probably don't override these
    @property
    def dims(self):
        return set(self.ishape).union(set(self.oshape))

    @property
    def H(self):
        """Adjoint operator"""
        if self._adj is None:
            _adj = copy(self)
            # Swap functions
            _adj.fn, _adj.adj_fn = self.adj_fn, self.fn
            _adj.split, _adj.adj_split = self.adj_split, self.split
            _adj.split_fn, adj.adj_split_fn = self.split_fn, self.adj_split_fn
            # Swap shapes
            _adj.ishape, _adj.oshape  = self.oshape, self.ishape
            _adj._suffix += '.H'
            self._adj = _adj
        return self._adj

    @property
    def N(self):
        """Normal operator (is this really necessary?)"""
        if self._normal is None:
        #     _normal = copy(self)
        #     _normal._suffix += '.N'
        #     self.normal = _normal
        # return self._normal
            _normal = copy(self)
            _normal.fn = self.normal_fn
            _normal.adj_fn = self.normal_fn
            def new_normal(x, *args, **kwargs):
                x = self.normal_fn(x, *args, **kwargs)
                return self.normal_fn(x, *args, **kwargs)
            _normal.normal_fn = new_normal
            _normal.ishape = self.ishape
            _normal.oshape, _normal.ishape = self.ishape, self.ishape
            _normal._suffix += '.N'
            self._normal = _normal
        return self._normal


    def split(self, ibatch, obatch):
        """Return a split version of the linop such that`forward`
        performs a split version of the linop
        ibatch: tuple of slices of same length as ishape
        obatch: tuple of slices of same length as oshape
        """
        return self.split_forward(ibatch, obatch)

    def adj_split(self, ibatch, obatch):
        """Split the adjoint version"""
        return self.split_forward(obatch, ibatch).H


    def split_fn(self, ibatch, obatch, /, **kwargs):
        """Return split versions of the data that can be passed
        into fn and adj_fn to produce split versions
        """
        return self.split_forward_fn(ibatch, obatch, **kwargs)

    def adj_split_fn(self, ibatch, obatch, /, **kwargs):
        return self.split_forward_fn(obatch, ibatch, **kwargs)

    def flatten(self):
        """Get a flattened list of constituent linops for composition"""
        return [self]

    def compose(self, inner):
        """Do self AFTER inner"""
        before = inner.flatten()
        after = self.flatten()
        return Chain(*(after + before))

    def __matmul__(self, other):
        return self.compose(other)

    def __rmatmul__(self, other):
        return other.compose(self)

    def __repr__(self):
        """Helps prevent recursion error caused by .H and .N"""
        return f'{self.__class__.__name__ + self._suffix}({self.ishape} -> {self.oshape})'


class Chain(NamedLinop):
    def __init__(self, *linops):
        super().__init__(linops[-1].ishape, linops[0].oshape)
        self.linops = list(linops)
        self.signatures = [signature(linop.fn) for linop in self.linops]
        self._check_signatures()
        self._check_inputs_outputs()

    def _check_signatures(self):
        seen = set()
        for sig in self.signatures:
            for param in sig.parameters.values():
                if param.name in seen:
                    logger.debug(
                        f'{param.name} appears more than once in linop chain.'
                    )

    def _check_inputs_outputs(self):
        curr_shape = self.ishape
        for linop in reversed(self.linops):
            if linop.ishape != curr_shape:
                raise ValueError(
                    f'Mismatched shape: expected {linop.ishape}, got {curr_shape} at input to {linop}'
                )
            curr_shape = linop.oshape

    def _parse_kwargs(self, kwargs):
        all_linop_kwargs = []
        for sig in self.signatures:
            linop_kwargs = {}
            for param in sig.parameters.values():
                if param.name in kwargs:
                    linop_kwargs[param.name] = kwargs[param.name]
            all_linop_kwargs.append(linop_kwargs)
        return all_linop_kwargs

    @property
    def H(self):
        """Adjoint operator"""
        if self._adj is None:
            linops = list(linop.H for linop in reversed(self.linops))
            _adj = type(self)(*linops)
            self._adj = _adj
        return self._adj

    @property
    def N(self):
        """Normal operator (is this really necessary?)"""
        if self._normal is None:
            linops = list(linop.H for linop in reversed(self.linops)) + self.linops
            _normal = type(self)(*linops)
            self._normal = _normal
        return self._normal

    def fn(self, x: torch.Tensor, /, *data_list):
        for linop, data in zip(reversed(self.linops), reversed(data_list)):
            x = linop.fn(x, data)
        return x

    def adj_fn(self, x: torch.Tensor, /, *data_list):
        for linop, data in zip(self.linops, data_list):
            x = linop.adj_fn(x, data)
        return x

    def normal_fn(self, x: torch.Tensor, /, *data_list):
        return self.adj_fn(self.fn(x, *data_list), *data_list)

    def flatten(self):
        return self.linops

    def split(self, *iobatches):
        """For compatibility with NamedLinop"""
        ibatches = iobatches[:len(iobatches)//2]
        obatches = iobatches[len(iobatches)//2:]
        return self.split_forward(ibatches, obatches)

    def adj_split(self, ibatches, obatches):
        ibatches = iobatches[:len(iobatches)//2]
        obatches = iobatches[len(iobatches)//2:]
        return self.split_forward(obatches, ibatches).H

    def split_forward(self, ibatches, obatches):
        """ibatches, obatches should correspond to the order in self.linops"""
        linops = [linop.split(ibatch, obatch) for ibatch, obatch, linop in zip(ibatches, obatches, self.linops)]
        return Chain(*linops)

    def split_data(self, kwargs_list):
        data = [linop.split_data(kwargs) for kwargs in kwargs_list]
        return data

    @property
    def dims(self):
        return set().union(*[linop.dims for linop in self.linops])

    def size(self, dim):
        for linop in self.linops:
            out = linop.size(dim)
            if out is not None:
                return out

    def size_fn(self, dim, **kwargs):
        all_linop_kwargs = self.parse_kwargs(kwargs)
        for linop, kw in zip(self.linops, all_linop_kwargs):
            out = linop.size_fn(dim, **kw)
            if out is not None:
                return out
        return None

    def __getitem__(self, idx):
        return self.linops[idx]

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        linop_chain = "\n\t".join(repr(linop) for linop in self.linops)
        return f'{self.__class__.__name__}(\n\t{linop_chain}\n)'
