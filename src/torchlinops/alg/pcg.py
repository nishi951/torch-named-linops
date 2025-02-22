from typing import Callable, Optional
from torch import Tensor

from dataclasses import dataclass

import torch
from tqdm import tqdm

from torchlinops.utils import default_to_dict, inner as zdot

__all__ = ["cg"]


def cg(
    A: Callable,
    y: Tensor,
    x0: Optional[Tensor] = None,
    max_num_iters: int = 20,
    gtol: float = 1e-3,
    ltol: float = 1e-5,
    disable_tracking: bool = False,
    tqdm_kwargs: Optional[dict] = None,
):
    """Solve Ax = y with conjugate gradients.

    A is a positive semidefinite matrix.



    Parameters
    ----------
    A : Callable
        Functional form of the (positive semidefinite) A matrix
    y : vector
        RHS of matrix equality
    x0 : Tensor, optional
        Initial guess at solution. If not provided, will initialize at zero
    max_num_iters : int, default 20
        The maximum number of iterations to run the algorithm
    tol : float, default 1e-6
        The relative change threshold for stopping early
    **tqdm_kwargs
        Extra keyword arguments to pass to tqdm

    Returns
    -------
    Tensor
        The result of conjugate gradient
    """
    # Default values
    if x0 is None:
        x = torch.zeros_like(y)
    else:
        x = x0.clone()
    tqdm_kwargs = default_to_dict(dict(desc="CG", leave=False), tqdm_kwargs)

    # Initialize run
    run = CGRun(ltol, gtol, A, y, disable=disable_tracking)
    run.update(x)

    r = y - A(x)
    p = r.clone()
    rs = zdot(r, r).real
    with tqdm(range(max_num_iters), **tqdm_kwargs) as pbar:
        for k in pbar:
            Ap = A(p)
            pAp = zdot(p, Ap)
            alpha = rs / pAp
            # Take step
            x = x + alpha * p
            r = r - alpha * Ap
            rs_old = rs.clone()
            rs = zdot(r, r).real
            run.update(x)
            # Stopping criterion
            if run.is_converged():
                break

            run.set_postfix(pbar)
            beta = rs / rs_old
            p = beta * p + r
    return run.x_out


@dataclass
class CGRun:
    """Track various aspects of the run

    Notes
    -----
    Assume A is positive definite.
    Letting A^(1/2) = B and z = B^(-H)y = B^(-1)y
    Then the solution to the least squares problem
        1/2||Bx - z||_2^2
    is given by the x satisfying
    B^HBx = B^Hz
    Ax = y <- The problem that CG solves

    For purposes of convergence testing, we can compute the gradients and loss values as
    grad = B^HBx - B^Hz
         = Ax - y
    loss = x^HB^HBx - x^HB^Hz - z^HBx + z^Hz
         = x^HAx - x^Hy - y^Hx + y^A^(-1)y



    """

    ltol: float
    gtol: float
    A: Callable
    y: Tensor
    x_out: Optional[Tensor] = None
    """The final tensor to output"""

    # Convergence
    prev_loss: float = None
    loss: float = float("inf")
    gnorm: float = None

    # Turn off tracking for speed
    disable: bool = False

    def update(self, x: Tensor):
        if self.disable:
            self.x_out = x
            return
        self.prev_loss = self.loss
        Ax = self.A(x)
        xy = zdot(x, self.y)
        self.loss = (zdot(x, Ax) - xy - xy.conj()).real.item()

        # Track best seen, or just update
        # if self.return_best:
        #     if self.loss < self.loss_best:
        #         self.x_out = x.clone()
        #         self.loss_best = self.loss
        # else:
        self.x_out = x

        # Compute grad norm
        grad = Ax - self.y
        self.gnorm = torch.linalg.vector_norm(grad).item()

    def set_postfix(self, pbar):
        if self.disable:
            return

        # Update progress bar if provided
        pbar.set_postfix(
            {
                "ldiff": abs(self.loss - self.prev_loss),
                "gnorm": self.gnorm,
            }
        )

    def is_converged(self) -> bool:
        if self.disable:
            return False
        ldiff = abs(self.loss - self.prev_loss)
        loss_converged = ldiff < self.ltol
        grad_converged = self.gnorm < self.gtol
        return loss_converged and grad_converged
