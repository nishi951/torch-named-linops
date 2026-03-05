import pytest
import torch

from torchlinops import Dense, Identity
from torchlinops.alg.poly import polynomial_preconditioner


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


def test_polynomial_preconditioner_high_degree():
    n = 5
    weight = torch.eye(n) * 2.0
    A = Dense(weight, ("N", "N"), ("N",), ("N",))

    P = polynomial_preconditioner(A, degree=5, norm="l_2", lower_eig=0.1, upper_eig=1.0)

    x = torch.randn(n)
    result = P(x)

    assert result.shape == x.shape
