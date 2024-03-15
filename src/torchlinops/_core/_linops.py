from copy import copy
from typing import Optional, Mapping

import torch
import torch.nn as nn
import torch.fft as fft
from einops import einsum, rearrange, reduce, repeat


__all__ = [
    "NamedLinop",
    "Chain",
    "Add",
    "Dense",
    "Diagonal",
    "FFT",
    "Repeat",
    "Identity",
    "Add",
    "Scalar",
]


class NamedLinop(nn.Module):
    """Base Class for all NamedLinops"""

    def __init__(self, ishape, oshape):
        """ishape and oshape are symbolic, not numeric
        They also change if the adjoint is taken (!)
        """
        super().__init__()
        self.ishape = ishape
        self.oshape = oshape

        self._adj = None
        self._normal = None

        self._suffix = ""

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
        raise NotImplementedError(f"{type(self).__name__} cannot be split.")

    # Override
    def split_forward_fn(self, ibatch, obatch, /, data=None):
        """Return data"""
        raise NotImplementedError(f"{type(self).__name__} cannot be split.")

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
            _adj.split_fn, _adj.adj_split_fn = self.split_fn, self.adj_split_fn
            # Swap shapes
            _adj.ishape, _adj.oshape = self.oshape, self.ishape
            _adj._suffix += ".H"
            self._adj = [_adj]  # Prevent registration as a submodule
        return self._adj[0]

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
            _normal._suffix += ".N"
            self._normal = [_normal]  # Prevent registration as a submodule
        return self._normal[0]

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

    def __add__(self, right):
        ...

    def __radd__(self, left):
        ...

    def __mul__(self, right):
        if isinstance(right, float) or isinstance(right, torch.Tensor):
            right = Scalar(weight=right, ioshape=self.ishape)
            return self.compose(right)
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, float) or isinstance(left, torch.Tensor):
            left = Scalar(weight=left, ioshape=self.oshape)
            return left.compose(self)
        return NotImplemented

    def __matmul__(self, right):
        return self.compose(right)

    def __rmatmul__(self, left):
        return left.compose(self)

    def __repr__(self):
        """Helps prevent recursion error caused by .H and .N"""
        return (
            f"{self.__class__.__name__ + self._suffix}({self.ishape} -> {self.oshape})"
        )


class Add(NamedLinop):
    def __init__(self, *linops):
        assert all(
            linop.ishape == linops[0].ishape for linop in linops
        ), "All linops must share same ishape"
        assert all(
            linop.oshape == linops[0].oshape for linop in linops
        ), "All linops must share same oshape"
        super().__init__(linops[0].ishape, linops[0].oshape)
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


class Broadcast(NamedLinop):
    """Return a rearrange matching batched ishape to oshape.
    Basically broadcast to each other
    TODO: Fix this class
    """

    def __init__(self, ishape, oshape):
        super().__init__(ishape, oshape)
        self.ishape_str = " ".join(ishape)
        self.oshape_str = " ".join(oshape)

    def forward(self, x):
        return self.fn(x, self.ishape_str, self.oshape_str)

    def fn(cls, x, ishape_str, oshape_str):
        return rearrange(x, f"... {ishape_str} -> ... {oshape_str}")

    def adj_fn(cls, x: torch.Tensor, ishape_str, oshape_str):
        return rearrange(x, f"... {oshape_str} -> ... {ishape_str}")

    def split(self, ibatch, obatch):
        return self  # Literally don't change anything


class Dense(NamedLinop):
    """
    Example:
    x: [A, Nx, Ny]
    weightshape: [A, T]
    oshape: [T, Nx, Ny]
    """

    def __init__(self, weight, weightshape, ishape, oshape):
        super().__init__(ishape, oshape)
        self.weight = weight
        self.weightshape = weightshape
        self.einstr = f'{" ".join(self.ishape)},{" ".join(self.weightshape)}->{" ".join(self.oshape)}'
        self.adj_einstr = f'{" ".join(self.oshape)},{" ".join(self.weightshape)}->{" ".join(self.ishape)}'

    def forward(self, x):
        return self.fn(x, self.weight)

    def fn(self, x, /, weight):
        return einsum(x, weight, self.einstr)

    def adj_fn(self, x, /, weight):
        return einsum(x, weight, self.adj_einstr)

    def normal_fn(self, x, /, weight):
        return self.adj_fn(self.fn(x, weight), weight)

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


class Diagonal(NamedLinop):
    def __init__(self, weight: torch.Tensor, ioshape):
        assert len(weight.shape) <= len(
            ioshape
        ), "All dimensions must be named or broadcastable"
        super().__init__(ioshape, ioshape)
        self.weight = weight

    def forward(self, x):
        return self.fn(x, self.weight)

    def fn(self, x, /, weight):
        return x * weight

    def adj_fn(self, x, /, weight):
        return x * torch.conj(weight)

    def normal_fn(self, x, /, weight):
        return x * torch.abs(weight) ** 2

    def split_forward(self, ibatch, obatch):
        weight = self.split_forward_fn(ibatch, obatch, self.weight)
        return type(self)(weight, self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, "Diagonal linop must be split identically"
        return weight[ibatch]

    def size(self, dim: str):
        return self.size_fn(dim, self.weight)

    def size_fn(self, dim: str, weight):
        if dim in self.ishape:
            return weight.shape[self.ishape.index(dim)]
        return None


class Identity(NamedLinop):
    def __init__(self, ioshape):
        super().__init__(ioshape, ioshape)

    def forward(self, x):
        return x

    def fn(self, x, /):
        return x

    def adj_fn(self, x, /):
        return x

    def normal_fn(self, x, /):
        return x

    def split_forward(self, ibatch, obatch):
        return type(self)(self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /):
        assert ibatch == obatch, "Identity linop must be split identically"
        return None

    def size(self, dim: str):
        return self.size_fn(dim)

    def size_fn(self, dim: str):
        return None


class FFT(NamedLinop):
    def __init__(self, ishape, oshape, dim, norm, centered: bool = False):
        """
        centered=True mimicks sigpy behavior
        """
        super().__init__(ishape, oshape)
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def forward(self, x, /):
        return self.fn(x)

    def fn(self, x):
        if self.centered:
            x = fft.ifftshift(x, dim=self.dim)
        x = fft.fftn(x, dim=self.dim, norm=self.norm)
        if self.centered:
            x = fft.fftshift(x, dim=self.dim)
        return x

    def adj_fn(self, x):
        if self.centered:
            x = fft.ifftshift(x, dim=self.dim)
        x = fft.ifftn(x, dim=self.dim, norm=self.norm)
        if self.centered:
            x = fft.fftshift(x, dim=self.dim)
        return x

    def normal_fn(self, x):
        return x

    def split_forward(self, ibatch, obatch):
        return self.split_forward_fn(ibatch, obatch)

    def split_forward_fn(self, ibatch, obatch, /):
        return type(self)(self.ishape, self.oshape, self.dim, self.norm)

    def size(self, dim):
        return self.size_fn(dim)

    def size_fn(self, dim: str, /):
        """FFT doesn't determine any dimensions"""
        return None


class Rearrange(NamedLinop):
    """Moves around dimensions."""

    def __init__(
        self, istr, ostr, ishape, oshape, axes_lengths: Optional[Mapping] = None
    ):
        assert len(ishape) == len(
            oshape
        ), "Rearrange currently only supports pure dimension permutations"
        super().__init__(ishape, oshape)
        self.istr = istr
        self.ostr = ostr
        self.axes_lengths = axes_lengths if axes_lengths is not None else {}

    def forward(self, x):
        return self.fn(x, self.istr, self.ostr, self.axes_lengths)

    def fn(self, x, /, istr, ostr, axes_lengths):
        return rearrange(x, f"{istr} -> {ostr}", **axes_lengths)

    def adj_fn(self, x, /, ostr, istr, axes_lengths):
        return rearrange(x, f"{ostr} -> {istr}", **axes_lengths)

    def split_forward(self, ibatch, obatch):
        """Rearranging is transparent to splitting"""
        return self

    def split_forward_fn(self, ibatch, obatch, /, istr, ostr, axes_lengths):
        """Rearranging is transparent to splitting"""
        return (istr, ostr, axes_lengths)

    def size(self, dim: str):
        """Rearranging does not determine any dimensions"""
        return None

    def size_fn(self, dim: str, /, istr, ostr, axes_lengths):
        """Rearranging does not determine any dimensions"""
        return None


class SumReduce(NamedLinop):
    """Wrapper for einops' reduce,

    Adjoint of Repeat
    """

    def __init__(self, ishape, oshape):
        """
        ipattern : string
            Input shape spec, einops style
        opattern : string
            Output shape spec, einops style
        """
        super().__init__(ishape, oshape)
        assert (
            len(self.oshape) < len(self.ishape)
        ), f"Reduce must be over at least one dimension: got {self.ishape} -> {self.oshape}"
        self.adj_ipattern = self.fill_singleton_dims(self.ishape, self.oshape)
        self.ipattern = " ".join(ishape)
        self.opattern = " ".join(oshape)

    @staticmethod
    def fill_singleton_dims(ishape, oshape):
        out = []
        for idim in ishape:
            if idim in oshape:
                out.append(idim)
            else:
                out.append("()")
        return out

    def forward(self, x):
        return self.fn(x, self.ipattern, self.opattern)

    def fn(self, x, /):
        x = reduce(x, f"{self.ipattern} -> {self.opattern}", "sum")
        return x

    def adj_fn(self, x, /):
        x = repeat(x, f"{self.opattern} -> {self.adj_ipattern}")
        return x

    def split_forward_fn(self, ibatch, obatch, /):
        """Reducing is transparent to splitting"""
        return tuple()

    def size(self, dim: str):
        """Reducing does not determine any dimensions"""
        return None

    def size_fn(self, dim: str, /, ipattern, opattern, size_spec):
        """Reducing does not determine any dimensions"""
        return None


class Repeat(NamedLinop):
    """Unsqueezes and expands a tensor along dim
    TODO: Replace with einops' repeat
    """

    def __init__(self, n_repeats, dim, ishape, oshape):
        assert len(ishape) + 1 == len(
            oshape
        ), "oshape should have 1 more dim than ishape"
        super().__init__(ishape, oshape)
        self.n_repeats = n_repeats
        self.dim = dim

    def forward(self, x):
        return self.fn(x, self.n_repeats)

    def fn(self, x, /, n_repeats):
        expand_size = [-1] * len(self.oshape)
        expand_size[self.dim] = n_repeats
        x = x.unsqueeze(self.dim)
        # print(x)
        return x.expand(*expand_size)

    def adj_fn(self, x, /, n_repeats):
        return torch.sum(x, dim=self.dim, keepdim=False)

    def split_forward(self, ibatch, obatch):
        """Repeat fewer times, depending on the size of obatch"""
        assert len(ibatch) == len(
            self.ishape
        ), "length of ibatch should match length of ishape"
        assert len(obatch) == len(
            self.oshape
        ), "length of obatch should match length of oshape"
        return type(self)(
            n_repeats=self.split_forward_fn(ibatch, obatch, self.n_repeats),
            dim=self.dim,
            ishape=self.ishape,
            oshape=self.oshape,
        )

    def split_forward_fn(self, ibatch, obatch, /, n_repeats):
        return self.slice_len(obatch[self.dim], n_repeats)

    def size(self, dim: str):
        return self.size_fn(dim, self.n_repeats)

    def size_fn(self, dim, /, n_repeats):
        if dim == self.oshape[self.dim]:
            return n_repeats
        return None

    @staticmethod
    def slice_len(slc, n):
        """
        n: length of sequence slc is being applied to
        """
        return len(range(*slc.indices(n)))


class Scalar(Diagonal):
    """The result of scalar multiplication

    A Diagonal linop that is trivially splittable.
    """

    def split_forward_fn(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, "Scalar linop must be split identically"
        return weight

    def size_fn(self, dim: str, weight):
        return None
