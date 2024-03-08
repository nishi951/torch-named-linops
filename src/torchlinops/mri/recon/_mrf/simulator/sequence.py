"""Interface to optimized-mrf"""
from dataclasses import dataclass
import numpy as np
import torch

from optim_mrf.epg_torch import GREParams, SteadyStateMRF, SequenceWrapper
from optim_mrf.initialize import tgas_load

__all__ = [
    "SequenceConfig",
    "load_sequence",
]


@dataclass
class SequenceConfig:
    nstates: int = 100
    """Number of EPG states"""
    real_signal: bool = True
    """Whether or not to use a real signal"""


def load_sequence(opt: SequenceConfig, device: torch.device):
    # Create sequence (customize this)
    flip_angle, TR, TE = tgas_load(mode="npz")
    # Inversion time; For inversion recovery
    TI: float = 15.0
    # Inversion angle in degrees; For inversion recovery:
    inv_angle: float = 180.0
    # Wait time; between first and second MRF groups
    wait_time: float = 1200.0

    sequence_len = len(flip_angle)
    fispparams = GREParams(
        flip_angle=flip_angle,
        flip_angle_requires_grad=False,
        TR=TR,
        TR_requires_grad=False,
        TE=TE,
        TE_requires_grad=False,
    )
    inv_rec_params = {
        "TI": torch.tensor(TI),
        "TI_requires_grad": False,
        "inv_angle": torch.tensor(inv_angle * np.pi / 180.0),
        "spoiler": True,
    }
    seq = SteadyStateMRF(fispparams, inv_rec_params, wait_time=torch.tensor(wait_time))
    seq = SequenceWrapper(seq, opt.nstates, opt.real_signal)
    seq.to(device)
    return seq
