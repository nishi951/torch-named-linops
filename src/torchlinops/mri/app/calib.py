from typing import Tuple, Optional, Mapping

import torch
import sigpy as sp
import numpy as np

from torchlinops.mri.calib.coil import get_mps_kgrid

class Calib:
    def __init__(
            self,
            trj: np.ndarray,
            ksp: np.ndarray,
            im_size: Tuple,
            calib_width: int = 24,
            kernel_width: int = 7,
            device: sp.Device = sp.cpu_device,
            espirit_kwargs: Optional[Mapping] = None,
    ):
        self.trj = trj
        self.ksp = ksp
        self.im_size = im_size
        self.calib_width = calib_width
        self.kernel_width = kernel_width
        self.device = device
        if espirit_kwargs is not None:
            self.espirit_kwargs = espirit_kwargs
        else:
            self.espirit_kwargs = {
                'thresh': 0.02,
                'max_iter': 100,
                'crop': 0.95,
            }

    def run(self):
        mps, kgrid = get_mps_kgrid(
            self.trj, self.ksp,
            self.im_size,
            calib_width=self.calib_width,
            kernel_width=self.kernel_width,
            device_idx=int(self.device),
            **self.espirit_kwargs
        )
        return mps, kgrid
