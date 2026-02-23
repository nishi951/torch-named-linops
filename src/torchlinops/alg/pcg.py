from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from torchlinops.utils import default_to_dict, inner as zdot

__all__ = ["conjugate_gradients"]


def conjugate_gradients(
    A: Callable,
    y: Tensor,
    x0: Optional[Tensor] = None,
    max_num_iters: int = 20,
    gtol: float = 1e-3,
    ltol: float = 1e-5,
    disable_tracking: bool = False,
    tqdm_kwargs: Optional[dict] = None,
) -> Tensor | None:
    """Solve $Ax = y$ with the conjugate gradient method.

    $A$ must be positive semidefinite (Hermitian). The algorithm iterates at
    most *max_num_iters* times or until both the loss-difference and
    gradient-norm convergence criteria are met.

    Parameters
    ----------
    A : Callable[[Tensor], Tensor]
        Function implementing the matrix-vector product $A(x)$.
    y : Tensor
        Right-hand side of the linear system.
    x0 : Tensor, optional
        Initial guess. Defaults to the zero vector.
    max_num_iters : int, default 20
        Maximum number of CG iterations.
    gtol : float, default 1e-3
        Convergence tolerance on the gradient norm $\\|Ax - y\\|$.
    ltol : float, default 1e-5
        Convergence tolerance on the absolute change in loss between
        successive iterations.
    disable_tracking : bool, default False
        If ``True``, skip loss/gradient tracking for speed (convergence
        checking is also disabled).
    tqdm_kwargs : dict, optional
        Extra keyword arguments forwarded to ``tqdm``.

    Returns
    -------
    Tensor or None
        The approximate solution $x$, or ``None`` if the solver was not
        able to produce a result.
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
    """Tracks convergence state during a conjugate gradient run.

    Assumes $A$ is positive definite. Monitors the quadratic loss
    $\\ell(x) = x^H A x - x^H y - y^H x + \\text{const}$ and the gradient
    norm $\\|Ax - y\\|$ to decide when to stop.

    Parameters
    ----------
    ltol : float
        Loss-difference convergence tolerance.
    gtol : float
        Gradient-norm convergence tolerance.
    A : Callable[[Tensor], Tensor]
        The linear operator.
    y : Tensor
        Right-hand side vector.
    x_out : Tensor, optional
        The current best iterate.
    prev_loss : float, optional
        Loss at the previous iteration.
    loss : float
        Loss at the current iteration.
    gnorm : float, optional
        Current gradient norm.
    disable : bool
        If ``True``, skip all tracking for speed.
    """

    ltol: float
    gtol: float
    A: Callable
    y: Tensor
    x_out: Optional[Tensor] = None

    # Convergence
    prev_loss: float = None
    loss: float = float("inf")
    gnorm: float = None

    # Turn off tracking for speed
    disable: bool = False

    def update(self, x: Tensor):
        """Update loss and gradient norm for iterate *x*.

        Parameters
        ----------
        x : Tensor
            The current CG iterate.
        """
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
        """Check whether both loss-difference and gradient-norm criteria are met."""
        if self.disable:
            return False
        ldiff = abs(self.loss - self.prev_loss)
        loss_converged = ldiff < self.ltol
        grad_converged = self.gnorm < self.gtol
        return loss_converged and grad_converged
