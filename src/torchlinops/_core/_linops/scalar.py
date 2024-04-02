import torch
import torch.nn as nn

from .diagonal import Diagonal


class Scalar(Diagonal):
    """The result of scalar multiplication

    A Diagonal linop that is trivially splittable.
    """

    def __init__(self, weight, ioshape, *args, **kwargs):
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)
        super().__init__(weight, ioshape, broadcast_dims=list(ioshape))

    def split_forward_fn(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, "Scalar linop must be split identically"
        return weight

    def size_fn(self, dim: str, weight):
        return None
