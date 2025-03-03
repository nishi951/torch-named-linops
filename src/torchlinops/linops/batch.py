from collections.abc import Callable
from typing import Union, Optional, Tuple
from torch import Tensor

import traceback
from pprint import pformat

import torch
import torch.nn as nn
from tqdm import tqdm

from .namedlinop import NamedLinop
from .identity import ShapeSpec
from .nameddim import ND, NS, Shape

from torchlinops.utils import batch_iterator, dict_product, INDENT

__all__ = ["Batch"]


class Batch(NamedLinop):
    def __init__(
        self,
        linop: NamedLinop,
        input_device: torch.device,
        output_device: torch.device,
        input_dtype: Union[str, torch.dtype],
        output_dtype: Union[str, torch.dtype],
        input_shape: Optional[Shape] = None,
        output_shape: Optional[Shape] = None,
        pbar: bool = False,
        name: Optional[str] = None,
        post_batch_hook: Optional[Callable] = None,
        **batch_sizes,
    ):
        """
        hook : Callable, optional
            Function that takes in the newly-created batch object and does stuff
        """
        # TODO: Should batch even have a shape???
        super().__init__(NS(linop.ishape, linop.oshape), name=name)

        self.linop = linop
        if input_shape is not None:
            self.linop = self.linop @ ShapeSpec(input_shape)
        if output_shape is not None:
            self.linop = ShapeSpec(output_shape) @ self.linop
        self.input_device = input_device
        self.output_device = output_device
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.pbar = pbar
        # self.name = name if name is not None else ""
        self.batch_sizes = batch_sizes
        self.post_batch_hook = post_batch_hook
        self.setup_batching()

    def setup_batching(self, hook: Optional[Callable] = None):
        self._linops = None
        self.batch_sizes = {ND.infer(k): v for k, v in self.batch_sizes.items()}
        self.sizes = self._precompute_sizes()
        _linops, self._input_batches, self._output_batches = self.make_tiles()
        self._linops = nn.ModuleList(_linops)
        self._shape = NS(self.linop.ishape, self.linop.oshape)
        super().reset_adjoint_and_normal()
        if self.post_batch_hook is not None:
            self.post_batch_hook(self)

    def to(self, device):
        self.input_device = device
        self.output_device = device
        self.linop.to(device)
        self.setup_batching()
        super().to(device)

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

        y = torch.zeros(
            tuple(self.sizes[dim] for dim in self.oshape),
            dtype=self.output_dtype,
            device=self.output_device,
        )
        for linop, in_batch, out_batch in tqdm(
            zip(self._linops, self._input_batches, self._output_batches),
            total=len(self._linops),
            desc=f"Batch({self.name}: {self.batch_sizes})",
            disable=(not self.pbar),
        ):
            try:
                xbatch = x[in_batch]
                ybatch = linop(xbatch)
                y[out_batch] += ybatch
            except RuntimeError:
                print(
                    f"linop: {linop}, in_batch: {in_batch}, out_batch: {out_batch}, self.batch_sizes: {self.batch_sizes}"
                )
                raise
        return y

    def make_tiles(self):
        """Construct, for each sub linop, the tile of input and output slices that it corresponds to."""
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
            input_batches.append(ibatches[0])  # Input batch of first linop
            output_batches.append(obatches[-1])  # Output batch of last linop
        return linops, input_batches, output_batches

    @property
    def H(self):
        if self._adjoint is None:
            try:
                _adjoint = self.adjoint()
                _adjoint._adjoint = [self]
                self._adjoint = [_adjoint]
            except AttributeError as e:
                traceback.print_exc()
                raise e
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
            post_batch_hook=self.post_batch_hook,
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
                raise e
        return self._normal[0]

    def normal(self, inner=None):
        normal_linop = self.linop.N
        # Collect shape updates from computing the normal
        shape_updates = getattr(normal_linop, "_shape_updates", {})
        new_batch_sizes = self.batch_sizes.copy()
        for d, nd in shape_updates.items():
            if d in self.batch_sizes:
                new_batch_sizes[shape_updates[d]] = self.batch_sizes[d]
        batch_size_kwargs = {str(k): v for k, v in new_batch_sizes.items()}
        normal = type(self)(
            linop=self.linop.N,
            input_device=self.input_device,
            output_device=self.input_device,
            input_dtype=self.input_dtype,
            output_dtype=self.input_dtype,
            name=self.name + ".N",
            pbar=self.pbar,
            post_batch_hook=self.post_batch_hook,
            **batch_size_kwargs,
        )

        return normal

    @staticmethod
    def fn(self, x, /, data):
        """TODO: Functional interface
        Specify data as a tuple of data entries, one for each linop in linops"""
        raise NotImplementedError(f"Batched functional interface not available yet.")
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
            xbatch = x[ibatches[0]].to(self.input_device)
            ybatch = self.linop.fn(xbatch, split_data)
            y[obatches[-0]] += ybatch
        return y

    @staticmethod
    def adj_fn(self, x, /, data):
        raise NotImplementedError("Batched linop has no adjoint (yet).")

    def size(self, dim):
        return self.linop.size(dim)

    def size_fn(self, dim, /, data=None):
        raise NotImplementedError()

    def __repr__(self):
        """Helps prevent recursion error caused by .H and .N"""
        output = ""
        output += INDENT.indent(self.name + self._suffix) + "(\n"
        with INDENT:
            output += repr(self.linop)
            output += ", "
            output += repr(pformat(self.batch_sizes)).strip("'")
            output += "\n"
        output += INDENT.indent(")")
        return output
