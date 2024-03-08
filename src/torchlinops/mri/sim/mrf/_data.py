from dataclasses import dataclass
from typing import Optional

import torch

from torchlinops.utils import Saveable

@dataclass
class SubspaceDataset(Saveable):
    trj: torch.Tensor
    mps: torch.Tensor
    ksp: torch.Tensor
    phi: torch.Tensor
    dic: Optional[torch.Tensor] = None
    img: Optional[torch.Tensor] = None
    field: Optional = None
