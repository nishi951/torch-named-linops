from dataclasses import dataclass
from typing import Optional, Mapping

import torch


@dataclass
class SubspaceDataset:
    trj: torch.Tensor
    """kspace trajectory"""
    mps: torch.Tensor
    """Sensitivity maps"""
    ksp: torch.Tensor
    """Simulated ksp data"""
    phi: torch.Tensor
    """Temporal Subspace"""
    q_img: Mapping[str, torch.Tensor]
    """Ground truth quantitative image"""
    t_img: Mapping[str, torch.Tensor]
    """Ground truth spatiotemporal image"""
    sub_img: Mapping[str, torch.Tensor]
    """Ground truth subspace image"""
    dic: Mapping[str, torch.Tensor]
    """Lookup dictionary"""
    t1: torch.Tensor
    """T1 values"""
    t2: torch.Tensor
    """T2 values"""
    field: Optional = None
    """Currently unused field object"""
