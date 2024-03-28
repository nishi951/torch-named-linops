import torch
import torch.nn as nn

from .diagonal import Diagonal


class Scalar(Diagonal):
    """The result of scalar multiplication

    A Diagonal linop that is trivially splittable.
    """
    def __init__(self, weight: float, ioshape):
        super(Diagonal, self).__init__(ioshape, ioshape)
        self.weight = nn.Parameter(torch.tensor(weight), requires_grad=False)
        assert (
            len(self.ishape) >= len(self.weight.shape)
        ), f"Weight cannot have fewer dimensions than the input shape: ishape: {self.ishape}, weight: {weight.shape}"

    def split_forward_fn(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, "Scalar linop must be split identically"
        return weight

    def size_fn(self, dim: str, weight):
        return None
