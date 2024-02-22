from typing import Tuple, Optional

import torch

from torchlinops.core import fake_dims
from torchlinops.core.linops import Diagonal
from torchlinops.mri.linops import NUFFT, SENSE
from torchlinops.mri.gridding.linops import GriddedNUFFT

from ...app.cg import ConjugateGradient

__all__= ['CGSENSE']

class CGSENSE:
    """Reconstruct a single image from kspace measurements"""
    def __init__(
            self,
            ksp: torch.Tensor,
            trj: torch.Tensor,
            mps: torch.Tensor,
            dcf: Optional[torch.Tensor] = None,
            num_iter: int = 20,
            gridded: bool = False
    ):
        self.ksp = ksp
        self.trj = trj
        self.mps = mps
        self.im_size = mps.shape[1:]
        self.dcf = dcf
        self.device = ksp.device
        if gridded:
            self.trj = self.trj.long()
            F = GriddedNUFFT(
                self.trj,
                self.im_size,
                in_batch_shape = ('C',),
                out_batch_shape = fake_dims('B', len(trj.shape) - 2),
            )
        else:
            F = NUFFT(
                self.trj,
                self.im_size,
                in_batch_shape = ('C',),
                out_batch_shape = fake_dims('B', len(trj.shape) - 2),
            )
        S = SENSE(self.mps)
        self.sense_linop = F @ S
        if self.dcf is not None:
            # Add density compensation
            D = Diagonal(
                torch.sqrt(self.dcf),
                ioshape=F.oshape,
            )
            self.sense_linop = D @ self.sense_linop
        AHb = self.sense_linop.H(ksp)
        self.cg_app = ConjugateGradient(
            A=self.sense_linop.N,
            b=AHb,
            x_init=AHb,
            num_iter=num_iter,
        )

    def run(self):
        return self.cg_app.run()
