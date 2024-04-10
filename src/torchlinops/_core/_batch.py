import traceback
from typing import Union, Optional
from pprint import pformat

import torch
from tqdm import tqdm

import torchlinops
from ._linops import NamedLinop, ND, NS
from torchlinops.utils import batch_iterator, dict_product

__all__ = ["Batch"]


class Batch(NamedLinop):
    def __init__(
        self,
        linop: NamedLinop,
        input_device: torch.device,
        output_device: torch.device,
        input_dtype: Union[str, torch.dtype],
        output_dtype: Union[str, torch.dtype],
        pbar: bool = False,
        name: Optional[str] = None,
        **batch_sizes,
    ):
        super().__init__(NS(linop.ishape, linop.oshape))
        self.linop = linop
        self.input_device = input_device
        self.output_device = output_device
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.pbar = pbar
        self.name = name if name is not None else ""
        self.batch_sizes = {ND.infer(k): v for k, v in batch_sizes.items()}
        self.sizes = self._precompute_sizes()
        self._linops, self._input_batches, self._output_batches = self.make_tiles()
        self.reset()

    def _precompute_sizes(self):
        sizes = {dim: self.linop.size(dim) for dim in self.linop.dims}
        return sizes

    @staticmethod
    def _make_batch_iterators(total_sizes, batch_sizes):
        batch_iterators = {}
        for dim, total in total_sizes.items():
            batch_iterators[dim] = (
                [slice(a, b) for a, b in batch_iterator(total, batch_sizes[dim])]
                if dim in batch_sizes
                else [slice(None)]
            )
        return batch_iterators

    def forward(self, x: torch.Tensor):
        # Complete the size specifications
        for dim, total in zip(self.ishape, x.shape):
            self.sizes[dim] = total
        # batch_iterators = self._make_batch_iterators(self.sizes, self.batch_sizes)
        # ishapes = [linop.ishape for linop in self.linop.flatten()]
        # oshapes = [linop.oshape for linop in self.linop.flatten()]

        y = torch.zeros(
            tuple(self.sizes[dim] for dim in self.oshape),
            dtype=self.output_dtype,
            device=self.output_device,
        )
        # for tile in tqdm(
        #     dict_product(batch_iterators),
        #     desc=f"Batch({self.batch_sizes})",
        #     disable=(not self.pbar),
        # ):
        for linop, in_batch, out_batch in tqdm(
            zip(self._linops, self._input_batches, self._output_batches),
            total=len(self._linops),
            desc=f"Batch({self.name}: {self.batch_sizes})",
            disable=(not self.pbar),
        ):
            # ibatches = [
            #     [tile.get(dim, slice(None)) for dim in ishape] for ishape in ishapes
            # ]
            # obatches = [
            #     [tile.get(dim, slice(None)) for dim in oshape] for oshape in oshapes
            # ]
            # linop = self.linop.split(self.linop, *ibatches, *obatches)
            xbatch = x[in_batch]
            ybatch = linop(xbatch)
            y[out_batch] += ybatch
        return y

    def make_tiles(self):
        # Complete the size specifications
        # for dim, total in zip(self.ishape, x.shape):
        #     self.sizes[dim] = total
        batch_iterators = self._make_batch_iterators(self.sizes, self.batch_sizes)
        ishapes = [linop.ishape for linop in self.linop.flatten()]
        oshapes = [linop.oshape for linop in self.linop.flatten()]
        tiles = list(dict_product(batch_iterators))
        linops = []
        input_batches = []
        output_batches = []
        for tile in tiles:
            ibatches = [
                [tile.get(dim, slice(None)) for dim in ishape] for ishape in ishapes
            ]
            obatches = [
                [tile.get(dim, slice(None)) for dim in oshape] for oshape in oshapes
            ]
            linop = self.linop.split(self.linop, *ibatches, *obatches)
            linops.append(linop)
            input_batches.append(ibatches[-1])
            output_batches.append(obatches[0])
        return linops, input_batches, output_batches

    def split_forward(self):
        # Complete the size specifications
        for dim, total in zip(self.ishape, x.shape):
            self.sizes[dim] = total
        batch_iterators = self._make_batch_iterators(self.sizes, self.batch_sizes)
        ishapes = [linop.ishape for linop in self.linop.flatten()]
        oshapes = [linop.oshape for linop in self.linop.flatten()]

        linops = {}
        for tile in tqdm(
            dict_product(batch_iterators),
            desc=f"Batch({self.batch_sizes})",
            disable=(not self.pbar),
        ):
            ibatches = [
                [tile.get(dim, slice(None)) for dim in ishape] for ishape in ishapes
            ]
            obatches = [
                [tile.get(dim, slice(None)) for dim in oshape] for oshape in oshapes
            ]
            linops[tile] = self.linop.split(*ibatches, *obatches)
        return linops

    @property
    def H(self):
        if self._adjoint is None:
            try:
                _adjoint = self.adjoint()
                _adjoint._adjoint = [self]
                self._adjoint = [_adjoint]
            except AttributeError as e:
                traceback.print_exc()
                raise
        return self._adjoint[0]

    def adjoint(self):
        batch_sizes = {str(k): v for k, v in self.batch_sizes.items()}
        adj = type(self)(
            linop=self.linop.H,
            input_device=self.output_device,
            output_device=self.input_device,
            input_dtype=self.output_dtype,
            output_dtype=self.input_dtype,
            name=self.name + ".H",
            pbar=self.pbar,
            **batch_sizes,
        )
        return adj

    @property
    def N(self):
        if self._normal is None:
            try:
                _normal = self.normal()
                self._normal = [_normal]
            except AttributeError as e:
                traceback.print_exc()
                raise
        return self._normal[0]

    def normal(self, inner=None):
        batch_sizes = {str(k): v for k, v in self.batch_sizes.items()}
        normal = type(self)(
            linop=self.linop.N,
            input_device=self.input_device,
            output_device=self.input_device,
            input_dtype=self.input_dtype,
            output_dtype=self.input_dtype,
            name=self.name + ".N",
            pbar=self.pbar,
            **batch_sizes,
        )
        return normal

    def fn(self, x, /, data):
        """TODO: Functional interface
        Specify data as a tuple of data entries, one for each linop in linops"""
        sizes = {}
        for dim in self.linop.dims:
            sizes[dim] = self.linop.size_fn(dim, data)
        for dim, total in zip(self.ishape, x.shape):
            sizes[dim] = total
        batch_iterators = self._make_batch_iterators(sizes, self.batch_sizes)
        ishapes = [linop.ishape for linop in self.linop.flatten()]
        oshapes = [linop.oshape for linop in self.linop.flatten()]

        y = torch.zeros(
            tuple(sizes[dim] for dim in self.oshape),
            dtype=self.output_dtype,
            device=self.output_device,
        )
        for tile in tqdm(
            dict_product(batch_iterators),
            desc=f"Batch({self.batch_sizes})",
            disable=(not self.pbar),
        ):
            ibatches = [
                [tile.get(dim, slice(None)) for dim in ishape] for ishape in ishapes
            ]
            obatches = [
                [tile.get(dim, slice(None)) for dim in oshape] for oshape in oshapes
            ]
            split_data = self.linop.split_fn(*ibatches, *obatches, *data)
            xbatch = x[ibatches[-1]].to(self.input_device)
            ybatch = self.linop.fn(xbatch, split_data)
            y[obatches[0]] += ybatch
        return y

    def adj_fn(self, x, /, data):
        raise NotImplementedError("Batched linop has no adjoint (yet).")

    def size(self, dim):
        return self.linop.size(dim)

    def size_fn(self, dim, /, data=None):
        raise NotImplementedError()

    def flatten(self):
        """Get a flattened list of constituent linops for composition
        Removes batching
        """
        return self.linop.flatten()

    def compose(self, inner):
        """Do self AFTER inner"""
        self.linop = self.linop.compose(inner)
        return self

    def __add__(self, right):
        self.linop = torchlinops.Add(self.linop, right)
        return self

    def __radd__(self, left):
        self.linop = torchlinops.Add(left, self.linop)
        return self

    def __mul__(self, right):
        if isinstance(right, float) or isinstance(right, torch.Tensor):
            right = torchlinops.Scalar(weight=right, ioshape=self.ishape)
            self.linop = self.linop.compose(right)
            return self
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, float) or isinstance(left, torch.Tensor):
            left = torchlinops.Scalar(weight=left, ioshape=self.oshape)
            self.linop = left.compose(self.linop)
            return self
        return NotImplemented

    def __matmul__(self, right):
        return self.compose(right)

    def __rmatmul__(self, left):
        return left.compose(self)

    def __repr__(self):
        """Helps prevent recursion error caused by .H and .N"""
        return f"{self.__class__.__name__ + self._suffix}(\n\t{self.linop}, {pformat(self.batch_sizes)})"
