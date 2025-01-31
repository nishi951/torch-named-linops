from typing import Optional

import torch
import torch.nn as nn

from torchlinops.utils import default_to

from .diagonal import Diagonal
from .nameddim import NDorStr, Shape


__all__ = ["Scalar"]


class Scalar(Diagonal):
    """The result of scalar multiplication

    A Diagonal linop that is trivially splittable.
    """

    def __init__(self, weight, ioshape: Optional[Shape] = None):
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)
        ioshape = default_to(("...",), ioshape)
        super().__init__(weight, ioshape=ioshape)

    def split_forward_fn(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, "Scalar linop must be split identically"
        return weight

    def size_fn(self, dim: str, weight):
        return None
