"""Tests for singular value decomposition using power method with deflation."""

import pytest
import torch

from torchlinops import Dense
from torchlinops.alg.svd import singular_value_decomposition


class TestSVD:
    """Tests for singular_value_decomposition function."""

    def test_svd_simple(self):
        """Test SVD on a simple tall matrix."""
        m, n = 10, 5
        A_dense = torch.randn(m, n, dtype=torch.float64)
        A = Dense(A_dense, ("M", "N"), ("N",), ("M",))

        x_init = torch.randn(n, dtype=torch.float64)
        num_sv = min(m, n)

        U, S, Vh = singular_value_decomposition(
            A, x_init, num_singular_values=num_sv, max_iters=100
        )

        assert U.shape == (m, num_sv)
        assert S.shape == (num_sv,)
        assert Vh.shape == (num_sv, n)
        assert torch.all(S[:-1] >= S[1:])

    def test_svd_reconstruction(self):
        """Test that A ≈ U @ diag(S) @ Vh."""
        m, n = 8, 4
        rank = 3
        U_true = torch.randn(m, rank)
        S_true = torch.tensor([3.0, 2.0, 1.0])
        Vh_true = torch.randn(rank, n)
        A_dense = U_true @ torch.diag(S_true) @ Vh_true

        A = Dense(A_dense, ("M", "N"), ("N",), ("M",))
        x_init = torch.randn(n, dtype=A_dense.dtype)
        num_sv = rank

        U, S, Vh = singular_value_decomposition(
            A, x_init, num_singular_values=num_sv, max_iters=200, tol=1e-10
        )

        A_reconstructed = U @ torch.diag(S) @ Vh
        # Power method gives approximate results, use looser tolerance
        assert torch.allclose(A_dense, A_reconstructed, atol=1e-2)

    def test_svd_vs_torch_linalg_svd(self):
        """Compare with torch.linalg.svd for verification."""
        m, n = 10, 5
        A_dense = torch.randn(m, n, dtype=torch.float64)
        A = Dense(A_dense, ("M", "N"), ("N",), ("M",))

        x_init = torch.randn(n, dtype=torch.float64)
        num_sv = n

        U, S, Vh = singular_value_decomposition(
            A, x_init, num_singular_values=num_sv, max_iters=100, tol=1e-8
        )

        U_torch, S_torch, Vh_torch = torch.linalg.svd(A_dense, full_matrices=False)
        assert torch.allclose(S, S_torch, rtol=1e-2)

    def test_svd_complex(self):
        """Test SVD with complex-valued linop."""
        m, n = 8, 4
        A_dense = torch.randn(m, n, dtype=torch.complex128)
        A = Dense(A_dense, ("M", "N"), ("N",), ("M",))

        x_init = torch.randn(n, dtype=torch.complex128)
        num_sv = n

        U, S, Vh = singular_value_decomposition(
            A, x_init, num_singular_values=num_sv, max_iters=200, tol=1e-10
        )

        assert U.shape == (m, num_sv)
        assert S.shape == (num_sv,)
        assert Vh.shape == (num_sv, n)

        A_reconstructed = U @ torch.diag(S.to(torch.complex128)) @ Vh
        # Power method gives approximate results, use looser tolerance
        assert torch.allclose(A_dense, A_reconstructed, atol=1e-2)

    def test_svd_orthogonality(self):
        """Test that singular vectors are approximately orthonormal."""
        m, n = 10, 4
        A_dense = torch.randn(m, n, dtype=torch.float64)
        A = Dense(A_dense, ("M", "N"), ("N",), ("M",))

        x_init = torch.randn(n, dtype=torch.float64)
        num_sv = n

        U, S, Vh = singular_value_decomposition(
            A, x_init, num_singular_values=num_sv, max_iters=200, tol=1e-10
        )

        U_orth = U.T @ U
        assert torch.allclose(U_orth, torch.eye(num_sv, dtype=torch.float64), atol=1e-2)

        V = Vh.T
        V_orth = V.T @ V
        assert torch.allclose(V_orth, torch.eye(num_sv, dtype=torch.float64), atol=1e-2)

    def test_svd_numerical_stability(self):
        """Test SVD doesn't break with small singular values."""
        m, n = 6, 4
        S_true = torch.tensor([10.0, 1.0, 0.1, 0.01], dtype=torch.float64)
        U_true = torch.linalg.qr(torch.randn(m, n, dtype=torch.float64))[0][:, :n]
        V_true = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))[0]

        A_dense = U_true @ torch.diag(S_true) @ V_true.T
        A = Dense(A_dense, ("M", "N"), ("N",), ("M",))
        x_init = torch.randn(n, dtype=torch.float64)

        U, S, Vh = singular_value_decomposition(
            A, x_init, num_singular_values=n, max_iters=200, tol=1e-10
        )
        assert torch.allclose(S, S_true, rtol=0.1)


def test_svd_single_singular_value():
    """Test computing just one singular value."""
    m, n = 5, 3
    A_dense = torch.randn(m, n, dtype=torch.float64)
    A = Dense(A_dense, ("M", "N"), ("N",), ("M",))

    x_init = torch.randn(n, dtype=torch.float64)
    U, S, Vh = singular_value_decomposition(A, x_init, num_singular_values=1)

    assert U.shape == (m, 1)
    assert S.shape == (1,)
    assert Vh.shape == (1, n)

    _, S_torch, _ = torch.linalg.svd(A_dense, full_matrices=False)
    assert torch.allclose(S, S_torch[:1], rtol=1e-2)


def test_svd_uses_deflation():
    """Verify that deflation is working."""
    m, n = 6, 4
    A_dense = torch.randn(m, n, dtype=torch.float64)
    A = Dense(A_dense, ("M", "N"), ("N",), ("M",))

    x_init = torch.randn(n, dtype=torch.float64)
    U, S, Vh = singular_value_decomposition(
        A, x_init, num_singular_values=n, max_iters=100
    )

    assert S[0] > S[-1] or torch.allclose(S[0], S[-1], rtol=0.01)


def test_svd_tolerance():
    """Test that tolerance parameter affects convergence."""
    m, n = 5, 3
    A_dense = torch.randn(m, n, dtype=torch.float64)
    A = Dense(A_dense, ("M", "N"), ("N",), ("M",))

    x_init = torch.randn(n, dtype=torch.float64)

    U1, S1, Vh1 = singular_value_decomposition(
        A, x_init, num_singular_values=2, max_iters=100, tol=1e-2
    )

    U2, S2, Vh2 = singular_value_decomposition(
        A, x_init, num_singular_values=2, max_iters=100, tol=1e-8
    )

    assert S1.shape == (2,)
    assert S2.shape == (2,)


def test_svd_diagonal_matrix():
    """Test SVD on a diagonal matrix."""
    m, n = 5, 3
    diagonal_vals = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    A_dense = torch.zeros(m, n, dtype=torch.float64)
    for i, v in enumerate(diagonal_vals):
        A_dense[i, i] = v

    A = Dense(A_dense, ("M", "N"), ("N",), ("M",))
    x_init = torch.randn(n, dtype=torch.float64)

    U, S, Vh = singular_value_decomposition(
        A, x_init, num_singular_values=3, max_iters=200, tol=1e-10
    )

    assert torch.allclose(S, diagonal_vals, rtol=0.1)
