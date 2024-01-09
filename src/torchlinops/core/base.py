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

    # Probably don't override these
    @property
    def H(self):
        """Adjoint operator"""
        if self._adj is None:
            _adj = copy(self)
            # Swap functions
            _adj.fn, _adj.adj_fn = self.adj_fn, self.fn
            _adj.split, _adj.adj_split = self.adj_split, self.split
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

    # Override these
    def fn(self, x: torch.Tensor, /, **kwargs):
        """Placeholder for functional forward operator.
        Non-input arguments should be keyword-only
        self can still be used - kwargs should contain elements
        that may change frequently (e.g. trajectories) and can
        ignore hyperparameters (e.g. normalization modes)
        """
        return x

    def adj_fn(self, x: torch.Tensor, /, **kwargs):
        """Placeholder for functional adjoint operator.
        Non-input arguments should be keyword-only"""
        return x

    def normal_fn(self, x: torch.Tensor, /, **kwargs):
        """Placeholder for efficient functional normal operator"""
        return self.adj_fn(self.fn(x, **kwargs), **kwargs)

    def split(self, ibatch, obatch):
        """Return a split version of the linop such that`forward`
        performs a split version of the linop
        ibatch: tuple of slices of same length as ishape
        obatch: tuple of slices of same length as oshape
        """
        return self._split(ibatch, obatch)

    def adj_split(self, ibatch, obatch):
        """Split the adjoint version"""
        return self._split(obatch, ibatch).H

    def _split(self, ibatch, obatch):
        """Split the forward version"""
        raise NotImplementedError(f'{self.__class__.__name__} cannot be split.')


    def _flatten(self):
        """Get a flattened list of constituent linops for composition"""
        return [self]

    def _compose(self, inner):
        """Do self AFTER inner"""
        before = inner._flatten()
        after = self._flatten()
        return Chain(*(after + before))

    def __matmul__(self, other):
        return self._compose(other)

    def __rmatmul__(self, other):
        return other._compose(self)

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

    def fn(self, x: torch.Tensor, /, **kwargs):
        all_linop_kwargs = self._parse_kwargs(kwargs)
        for linop, kw in zip(reversed(self.linops),
                             reversed(all_linop_kwargs)):
            x = linop(x, **kw)
        return x

    def adj_fn(self, x: torch.Tensor, /, **kwargs):
        all_linop_kwargs = self._parse_kwargs(kwargs)
        for linop, kw in zip(self.linops, all_linop_kwargs):
            x = linop.adj_fn(x, **kw)
        return x

    def normal_fn(self, x: torch.Tensor, /, **kwargs):
        return self.adj_fn(self.fn(x, **kwargs))

    def _flatten(self):
        return self.linops

    def __repr__(self):
        linop_chain = "\n\t".join(repr(linop) for linop in self.linops)
        return f'{self.__class__.__name__}(\n\t{linop_chain}\n)'
