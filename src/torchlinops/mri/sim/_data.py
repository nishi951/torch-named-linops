from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["MRIDataset"]


@dataclass
class MRIDataset:
    trj: torch.Tensor
    mps: torch.Tensor
    ksp: torch.Tensor
    img: Optional[torch.Tensor] = None
    field: Optional = None
