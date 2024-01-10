from einops import rearrange
import torch

from torchlinops.utils import batch_iterator, dict_product

class Batch(NamedLinop):
    def __init__(self, linop, **batch_sizes):
        self.linop = linop
        self.batch_sizes = batch_sizes
        super().__init__(self.linop.ishape, self.linop.oshape)
        self.sizes = self._precompute_sizes()

    def _precompute_sizes(self):
        sizes = {dim: self.linop.size(dim) for dim in self.linop.dims}
        return sizes

    @staticmethod
    def _make_batch_iterators(total_sizes, batch_sizes):
        batch_iterators = {}
        for dim, total in total_sizes.items():
            batch_iterators[dim] = batch_iterator(total, batch_sizes[dim])
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
            y[obatches[-1]] += linop(x)
        return y

    def fn(self, x, /, **data):
        sizes = {}
        for dim, total in zip(self.ishape, x.shape):
            sizes[dim] = total
        batch_iterators = self._make_batch_iterators(sizes, self.batch_sizes)
        ishapes = [linop.ishape for linop in self.linop.flatten()]
        oshapes = [linop.oshape for linop in self.linop.flatten()]

        y = torch.zeros(tuple(sizes[dim] for dim in self.oshape), dtype=x.dtype)
        for tile in dict_product(batch_iterators):
            split_data = self.linop.split_data(all_linop_kwargs)
            kw = linop.split_fn(data)
            x = linop.fn(x, **kw)
        return x
