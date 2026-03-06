import numpy as np
import pytest
import torch

from torchlinops import Dense, Identity
from torchlinops.alg.poly import ifista_coeffs, l_2_opt, polynomial_preconditioner


def test_polynomial_preconditioner_degree_minus_one():
    n = 5
    weight = torch.eye(n)
    A = Dense(weight, ("N", "N"), ("N",), ("N",))

    result = polynomial_preconditioner(A, degree=-1)

    assert isinstance(result, Identity)


def test_polynomial_preconditioner_degree_zero():
    n = 5
    weight = torch.eye(n)
    A = Dense(weight, ("N", "N"), ("N",), ("N",))

    result = polynomial_preconditioner(A, degree=0)

    assert result is not None


def test_polynomial_preconditioner_l2_norm():
    n = 5
    weight = torch.eye(n) * 2.0
    A = Dense(weight, ("N", "N"), ("N",), ("N",))

    P = polynomial_preconditioner(A, degree=2, norm="l_2", lower_eig=0.1, upper_eig=1.0)

    assert P is not None


@pytest.mark.skip(reason="Requires Chebyshev module")
def test_polynomial_preconditioner_l_inf_norm():
    n = 5
    weight = torch.eye(n) * 2.0
    A = Dense(weight, ("N", "N"), ("N",), ("N",))

    P = polynomial_preconditioner(
        A, degree=2, norm="l_inf", lower_eig=0.1, upper_eig=1.0
    )

    assert P is not None


def test_polynomial_preconditioner_ifista_norm():
    n = 5
    weight = torch.eye(n) * 2.0
    A = Dense(weight, ("N", "N"), ("N",), ("N",))

    P = polynomial_preconditioner(
        A, degree=2, norm="ifista", lower_eig=0.1, upper_eig=1.0
    )

    assert P is not None


def test_polynomial_preconditioner_invalid_norm():
    n = 5
    weight = torch.eye(n)
    A = Dense(weight, ("N", "N"), ("N",), ("N",))

    with pytest.raises(ValueError):
        polynomial_preconditioner(A, degree=2, norm="invalid_norm")


def test_polynomial_preconditioner_apply():
    n = 5
    weight = torch.eye(n) * 2.0
    A = Dense(weight, ("N", "N"), ("N",), ("N",))

    P = polynomial_preconditioner(A, degree=1, norm="l_2", lower_eig=0.1, upper_eig=1.0)

    x = torch.randn(n)
    result = P(x)

    assert result.shape == x.shape


def test_polynomial_preconditioner_high_degree():
    n = 5
    weight = torch.eye(n) * 2.0
    A = Dense(weight, ("N", "N"), ("N",), ("N",))

    P = polynomial_preconditioner(A, degree=5, norm="l_2", lower_eig=0.1, upper_eig=1.0)

    x = torch.randn(n)
    result = P(x)

    assert result.shape == x.shape


# --- Direct tests for l_2_opt and ifista_coeffs ---


def test_l_2_opt_returns_coefficients():
    """l_2_opt should return (ndarray, sympy expr) with degree+1 coefficients."""
    coeffs, poly = l_2_opt(degree=1, lower=0.1, upper=1.0, verbose=False)
    assert len(coeffs) == 2  # degree 1 -> 2 coefficients
    assert isinstance(coeffs[0], np.floating)


def test_l_2_opt_degree_zero():
    """Degree-0 polynomial should return a single coefficient."""
    coeffs, poly = l_2_opt(degree=0, lower=0.1, upper=1.0, verbose=False)
    assert len(coeffs) == 1


def test_ifista_coeffs_returns_correct_length():
    """ifista_coeffs(degree) should return degree+1 coefficients."""
    coeffs = ifista_coeffs(degree=3)
    assert len(coeffs) == 4


def test_ifista_coeffs_degree_one():
    """ifista_coeffs(1) = [binom(2,1)*(-1)^0, binom(2,2)*(-1)^1] = [2, -1]."""
    coeffs = ifista_coeffs(degree=1)
    assert coeffs[0] == pytest.approx(2.0)
    assert coeffs[1] == pytest.approx(-1.0)
