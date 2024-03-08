import torch

from ._linops import NamedLinop
from torchlinops.utils import batch_iterator, dict_product

__all__ = ['Batch']

class Batch(NamedLinop):
    """TODO:"""
    def __init__(self, linop, **batch_sizes):
        super().__init__(linop.ishape, linop.oshape)
        self.linop = linop
        self.batch_sizes = batch_sizes
        self.sizes = self._precompute_sizes()

    def _precompute_sizes(self):
        sizes = {dim: self.linop.size(dim) for dim in self.linop.dims}
        return sizes

    @staticmethod
    def _make_batch_iterators(total_sizes, batch_sizes):
        batch_iterators = {}
        for dim, total in total_sizes.items():
            batch_iterators[dim] = [slice(a, b) for a, b in batch_iterator(total, batch_sizes[dim])] \
                if dim in batch_sizes else [slice(None)]
        return batch_iterators

    def forward(self, x: torch.Tensor):
        # Complete the size specifications
        for dim, total in zip(self.ishape, x.shape):
            self.sizes[dim] = total
        batch_iterators = self._make_batch_iterators(self.sizes, self.batch_sizes)
        ishapes = [linop.ishape for linop in self.linop.flatten()]
        oshapes = [linop.oshape for linop in self.linop.flatten()]

        y = torch.zeros(tuple(self.sizes[dim] for dim in self.oshape), dtype=x.dtype)
        for tile in dict_product(batch_iterators):
            # TODO: Make this more efficient
            ibatches = [[tile.get(dim, slice(None)) for dim in ishape] for ishape in ishapes]
            obatches = [[tile.get(dim, slice(None)) for dim in oshape] for oshape in oshapes]
            linop = self.linop.split(*ibatches, *obatches)
            ybatch = linop(x)
            y[obatches[0]] += ybatch
        return y

    def fn(self, x, /, data):
        """Specify data as a tuple of data entries, one for each linop in linops"""
        sizes = {}
        for dim in self.linop.dims:
            sizes[dim] = self.linop.size_fn(dim, data)
        for dim, total in zip(self.ishape, x.shape):
            sizes[dim] = total
        batch_iterators = self._make_batch_iterators(sizes, self.batch_sizes)
        ishapes = [linop.ishape for linop in self.linop.flatten()]
        oshapes = [linop.oshape for linop in self.linop.flatten()]

        y = torch.zeros(tuple(sizes[dim] for dim in self.oshape), dtype=x.dtype)
        for tile in dict_product(batch_iterators):
            ibatches = [[tile.get(dim, slice(None)) for dim in ishape] for ishape in ishapes]
            obatches = [[tile.get(dim, slice(None)) for dim in oshape] for oshape in oshapes]
            split_data = self.linop.split_fn(*ibatches, *obatches, *data)
            ybatch = self.linop.fn(x, split_data)
            y[obatches[0]] += ybatch
        return y

    def adj_fn(self, x, /, data):
        raise NotImplementedError('Batched linop has no adjoint (yet).')
