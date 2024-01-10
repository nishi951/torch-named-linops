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
        for dim in self.linop.dims():
            total = self.linop.size(dim)
            if total is not None:
                batch_totals[dim] = total
        return batch_totals

    def _make_batch_iterators(self):
        batch_iterators = {}
        for dim, total in self.batch_totals.items():
            batch_iterators[dim] = batch_iterator(total, self.batch_sizes[dim])
        return batch_iterators

    def forward(self, x: torch.Tensor):
        for dim, total in zip(self.ishape, x.shape):
            self.batch_totals[dim] = total
        batch_iterators = self._make_batch_iterators()

        y =
        for tile in dict_product(batch_iterators):
            for linop in reversed(self.linop._flatten()):
                ibatch = [tile.get(dim, slice(None)) for dim in linop.ishape]
                obatch = [tile.get(dim, slice(None)) for dim in linop.oshape]
                split_linop = linop.split(ibatch, obatch)
                x = split_linop(x)
            # Combine everything


        return x
