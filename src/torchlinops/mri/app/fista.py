from typing import Optional, Callable

import torch

from torchlinops.core.base import NamedLinop
from torchlinops.mri.recon.powermethod import PowerMethod
from torchlinops.mri.recon.fista import (
    FISTA as fista_obj,
    FISTAHparams as fista_config,
)


class FISTA:
    def __init__(
            self,
            A: NamedLinop,
            b: torch.Tensor,
            prox: Callable,
            num_iters: int = 40,
            max_eig: Optional[float] = None,
            max_eig_iters: Optional[int] = 30,
            precond: Optional[Callable] = None,
            log_every: int = 1,
            state_every: int = 1,
    ):
        device = b.device
        if max_eig is None:
            power_method = PowerMethod(num_iter=max_eig_iters)
            max_eig = power_method(A.N, ishape, device)
        self.A = 1./torch.sqrt(max_eig) * A

        self.fista_module = fista_obj(
            self.A,
            self.A.H,
            prox,
            fista_config(
                lr=1.,
                num_iters=num_iters,
                log_every=log_every,
                state_every=state_every,
            ),
            self.A.N,
            precond,
        )

        self.b = b

    def run(self):
        """
        Note:
        Other states are accessible through self.fista_module.logs
        or self.fista_module.states
        """

        s = self.fista_module.run(self.b)
        return s.z
