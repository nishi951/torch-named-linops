from dataclasses import dataclass
from typing import Optional

import torch

from torchlinops.utils import Saveable

__all__ = ["MRIDataset"]


@dataclass
class MRIDataset(Saveable):
    trj: torch.Tensor
    mps: torch.Tensor
    ksp: torch.Tensor
    img: Optional[torch.Tensor] = None
    field: Optional = None
