from torchlinops.mri.recon.fista import FISTA as fista_obj, FISTAHparams as fista_config

class FISTA:
    def __init__(
            self,
            A: NamedLinop,
            b: torch.Tensor,
            prox: Callable,
            num_iters: int = 40,
            max_eig: Optional[float] = None,
            max_eig_iters: Optional[int] = 30,
            log_every: int = 1,
            state_every: int = 1,
    ):
        if max_eig is None:
            power_method = PowerMethod(num_iter=max_eig_iters)
            max_eig = power_method(A.N, ishape, device)
        self.A = 1./torch.sqrt(max_eig) * A


    def run(self):
        ...
