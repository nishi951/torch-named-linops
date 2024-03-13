from typing import Optional
import torch

from torchlinops.core.linops import NamedLinop
from torchlinops.mri.recon.pcg import CGHparams as cg_hparams, ConjugateGradient as cg

__all__ = ["ConjugateGradient"]


class ConjugateGradient:
    """App version of ConjugateGradient algorithm
    Approximately solves the equation

    Ax = b

    where A is positive semidefinite.

    """

    def __init__(
        self,
        A: NamedLinop,
        b: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        num_iter: int = 20,
    ):
        """
        Any device-related stuff should be taken care of
        beforehand.
        Any preconditioning should be incorporated into the linear operator.
        """
        self.A = A
        self.b = b
        if x_init is None:
            b_shape = self.A.H(b)
            x_init = torch.zeros_like(b)
        else:
            self.x_init = x_init

        self.cg_hparams = cg_hparams(num_iter)
        self.cg = cg(A, self.cg_hparams)

    def run(self) -> torch.Tensor:
        """Run the actual conjugate gradient algorithm.

        Returns
        -------
        torch.Tensor : The estimate of x
        """
        return self.cg(y=self.b, x_init=self.x_init)
