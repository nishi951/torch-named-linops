from dataclasses import dataclass, field, asdict
from typing import Optional, Literal

import numpy as np

__all__ = ['SimulatorConfig', 'Simulator']


@dataclass
class SimulatorConfig:
    nufft: dict = field(default_factory={
        'oversamp': None,
        'width': None,
    })
    dcf: dict = field(default_factory={
        'max_iter': 30,
    })

class Simulator:
    def __init__(
            self,
            sequence,
            trj,
            im_size,
            params: SimulatorConfig,
            device_idx: int = -1,
    ):

        self.params = params
        self.sequence = sequence
        self.trj = trj
        self.im_size = im_size

    def simulate(self, phantom: np.ndarray, mps: np.ndarray):
        """
        C: Number of coils
        Q: Quantitative parameters dimension
        im_size: Image size

        phantom: [Q *im_size]
        mps: [C *im_size]
        """
        assert len(mps.shape) == len(phantom.shape)
        assert mps.shape[1:] == phantom.shape[1:]
        assert self.im_size == mps.shape[1:]
        # Simulate sequence
        per_voxel_signal = self.sequence.run(phantom)
        # Apply sensitivity maps
        weighted_voxel_signal = mps * per_voxel_signal
        # Fourier transform
        ksp = sp.nufft(weighted_voxel_signal, self.trj, **self.params.nufft)
        return ksp
