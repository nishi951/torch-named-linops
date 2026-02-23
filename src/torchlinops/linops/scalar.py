from typing import Optional

import torch

from torchlinops.utils import default_to

from .diagonal import Diagonal
from ..nameddim import Shape

__all__ = ["Scalar"]


class Scalar(Diagonal):
    """Scalar multiplication operator $S(x) = \\alpha x$.

    A special case of ``Diagonal`` where the weight is a scalar, making it
    trivially splittable (the same scalar applies to every tile).
    """

    def __init__(self, weight, ioshape: Optional[Shape] = None):
        """
        Parameters
        ----------
        weight : float or Tensor
            The scalar multiplier $\\alpha$.
        ioshape : Shape, optional
            Named dimensions (same for input and output).
        """
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)
        ioshape = default_to(("...",), ioshape)
        super().__init__(weight, ioshape=ioshape)

    def split_weight(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, "Scalar linop must be split identically"
        return weight

    def size(self, dim: str):
        return None
