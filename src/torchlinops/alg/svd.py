from typing import Optional, Tuple

import torch
from torch import Tensor

from torchlinops.linops.dense import Dense
from torchlinops.alg.powermethod import power_method
from torchlinops.utils import default_to_dict

__all__ = ["singular_value_decomposition"]


def singular_value_decomposition(
    A: Dense,
    x_init: Tensor,
    num_singular_values: int,
    max_iters: int = 50,
    tol: float = 1e-5,
    eps: float = 0.0,
    dim: Optional[int | Tuple[int, ...]] = None,
    tqdm_kwargs: Optional[dict] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute compact SVD of a "tall" linop using power method with deflation.

    For a tall linear operator A (m x n with m >= n), computes the singular
    value decomposition A ≈ U @ diag(S) @ Vh using iterative deflation.

    The algorithm:

    1. Compute A.N = A^H A (the normal operator)
    2. Find largest eigenvalue λ and eigenvector v of A.N via power iteration
    3. Singular value: σ = √λ
    4. Left singular vector: u = A(v) / σ
    5. Deflate: A.N ← A.N - σ² v v^H
       - Note: v v^H = v.H.N
    6. Repeat for subsequent singular values

    Parameters
    ----------
    A : Dense
        Tall linear operator with shape (m, n) where m >= n.
        Currently supports Dense linops for constructing deflation operators.
    x_init : Tensor
        Initial vector for power iteration. Determines the input shape
        and dtype. For batched computation, includes batch dimensions.
    num_singular_values : int
        Number of singular values and vectors to compute.
    max_iters : int, default 50
        Maximum power iterations per singular value.
    tol : float, default 1e-5
        Convergence tolerance for eigenvalue estimation.
    eps : float, default 0.0
        Small constant for numerical stability in normalization.
    dim : int or tuple of int, optional
        Batch dimension(s) for batched power iteration.
    tqdm_kwargs : dict, optional
        Keyword arguments forwarded to tqdm progress bar.

    Returns
    -------
    U : Tensor
        Left singular vectors. Shape: (*oshape, num_singular_values)
        where oshape is A's output shape.
    S : Tensor
        Singular values in descending order. Shape: (num_singular_values,)
    Vh : Tensor
        Conjugate transpose of right singular vectors.
        Shape: (num_singular_values, *ishape) where ishape is A's input shape.

    Examples
    --------
    >>> A = Dense(torch.randn(10, 5), ("M", "N"), ("N",), ("M",))
    >>> x_init = torch.randn(5)
    >>> U, S, Vh = singular_value_decomposition(A, x_init, num_singular_values=3)
    >>> # Verify: A @ x ≈ U @ diag(S) @ Vh @ x for reconstruction
    """
    tqdm_kwargs = default_to_dict(dict(desc="SVD"), tqdm_kwargs)

    ishape = A.ishape

    # Storage for singular vectors and values
    U_list = []
    S_list = []
    V_list = []

    # Current deflated normal operator
    current_op = A.N

    # Track if we're in complex domain
    is_complex = x_init.is_complex()

    for i in range(num_singular_values):
        # Update progress bar description
        iter_desc = f"{tqdm_kwargs.get('desc', 'SVD')} [σ_{i + 1}]"
        iter_tqdm_kwargs = {**tqdm_kwargs, "desc": iter_desc}

        # Power iteration to find largest eigenvalue/eigenvector
        v, eigenvalue = power_method(
            current_op,
            x_init.clone(),
            max_iters=max_iters,
            tol=tol,
            eps=eps,
            dim=dim,
            tqdm_kwargs=iter_tqdm_kwargs,
        )  # Singular value is sqrt of eigenvalue
        sigma = torch.sqrt(eigenvalue)

        # Compute left singular vector: u = A(v) / sigma
        u = A(v) / (sigma + eps)

        # Store results
        U_list.append(u)
        S_list.append(sigma.squeeze() if sigma.dim() > 0 else sigma)
        V_list.append(v)

        # Create a unique singleton dimension name for this iteration
        singleton_dim = f"svd_s{i}"

        weight = v.unsqueeze(-1)

        # Create Dense for the rank-1 operator
        ishape_tuple = tuple(str(d) for d in ishape)
        rank1_weightshape = ishape_tuple + (singleton_dim,)
        rank1_ishape = (singleton_dim,)
        rank1_oshape = ishape_tuple

        V = Dense(
            weight,
            weightshape=rank1_weightshape,
            ishape=rank1_ishape,
            oshape=rank1_oshape,
        )

        current_op = current_op - eigenvalue * V.H.N

    # Stack results
    # U: (*oshape, num_singular_values)
    # S: (num_singular_values,)
    # V: (*ishape, num_singular_values)
    U = torch.stack(U_list, dim=-1)
    S = torch.stack(S_list)
    V = torch.stack(V_list, dim=-1)

    # Vh is the conjugate transpose of V
    # Current V shape: (*ishape, num_singular_values)
    # Target Vh shape: (num_singular_values, *ishape)
    if is_complex:
        Vh = V.conj()
    else:
        Vh = V

    # Move num_singular_values dimension to front
    # From (*ishape, num_singular_values) to (num_singular_values, *ishape)
    num_sv_dim = -1  # Last dimension
    Vh = Vh.transpose(num_sv_dim, 0)
    for i in range(1, len(ishape)):
        Vh = Vh.transpose(i - 1, i)

    return U, S, Vh
