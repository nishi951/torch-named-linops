from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from optim_mrf.epg_torch.mag_prep import (
    InversionRecoveryConfig,
)
from optim_mrf.epg_torch.sequences import (
    SteadyStateMRF,
    GREParams as GREConfig,
    SequenceWrapper,
)


@dataclass
class SteadyStateMRFSimulatorConfig:
    fisp_config: GREConfig
    inv_rec_config: InversionRecoveryConfig
    wait_time: float
    """Waiting time between first and second MRF sequences"""
    num_states: int
    real_signal: bool


class SteadyStateMRFSimulator(nn.Module):
    """Simulates two FISP sequences with the same flip angle train
    Outputs the second (steady-state) signal levels for the given (PD, T1, T2)
    input
    """

    def __init__(self, config: SteadyStateMRFSimulatorConfig):
        super().__init__()
        self.config = config

        seq = SteadyStateMRF(
            self.config.fisp_config,
            asdict(self.config.inv_rec_config),
            wait_time=torch.tensor(self.config.wait_time),
        )
        self.seq = SequenceWrapper(seq, self.config.num_states, self.config.real_signal)

    def forward(self, *phantom_t1t2pd):
        signal = self.seq(*phantom_t1t2pd)
        return signal
