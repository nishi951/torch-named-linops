from copy import copy

import pytest
import torch

from torchlinops import Diagonal
from torchlinops.tests.test_base import BaseNamedLinopTests
from torchlinops.utils import inner


class TestDiagonal(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        M = 10
        weight = torch.randn(M, 1, 1, dtype=torch.complex64)
        ioshape = ("M", "N", "P")
        A = Diagonal(weight, ioshape)
        N, P = 5, 7
        x = torch.randn(M, N, P, dtype=torch.complex64)
        y = torch.randn(M, N, P, dtype=torch.complex64)
        return A, x, y


@pytest.mark.xfail  # Deprecated behavior: changing a non-() dim to ()
def test_diagonal_shape_renaming():
    M = 10
    weight = torch.randn(M, 1, 1, dtype=torch.complex64)
    A = Diagonal(weight, ("M", "N", "P"))
    B = copy(A)
    new_ioshape = ("()", "N1", "P1")
    B.oshape = new_ioshape
    assert B.ishape == new_ioshape

    new_ioshape = ("M1", "N1", "P1")
    B.oshape = new_ioshape
    assert B.ishape == new_ioshape


def test_diagonal_pow():
    weight = torch.randn(5, dtype=torch.complex64)
    A = Diagonal(weight, ("N",))
    A2 = A**2
    x = torch.randn(5, dtype=torch.complex64)
    assert torch.allclose(A2(x), x * weight**2)


class TestDiagonalFromWeight(BaseNamedLinopTests):
    """Test Diagonal.from_weight as a BaseNamedLinopTests subclass."""

    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        # weight has shape (C, Nx) and ioshape is (C, Nx, Ny)
        C, Nx, Ny = 3, 4, 5
        weight = torch.randn(C, Nx, dtype=torch.complex64)
        A = Diagonal.from_weight(
            weight,
            weight_shape=("C", "Nx"),
            ioshape=("C", "Nx", "Ny"),
            shape_kwargs={"Ny": Ny},
        )
        x = torch.randn(C, Nx, Ny, dtype=torch.complex64)
        y = torch.randn(C, Nx, Ny, dtype=torch.complex64)
        return A, x, y


def test_diagonal_from_weight_too_many_dims():
    """from_weight should raise ValueError when weight has more dims than ioshape."""
    weight = torch.randn(3, 4, 5, dtype=torch.complex64)
    with pytest.raises(ValueError, match="dimensions must be named"):
        Diagonal.from_weight(
            weight, weight_shape=("C", "Nx", "Ny"), ioshape=("C", "Nx")
        )


def test_diagonal_from_weight_matches_direct():
    """from_weight result should numerically match direct construction."""
    C, N = 3, 5
    weight = torch.randn(C, N, dtype=torch.complex64)
    A_from = Diagonal.from_weight(weight, weight_shape=("C", "N"), ioshape=("C", "N"))
    A_direct = Diagonal(weight, ("C", "N"))
    x = torch.randn(C, N, dtype=torch.complex64)
    assert torch.allclose(A_from(x), A_direct(x))


def test_diagonal_split_with_broadcast_dims():
    """Splitting a Diagonal with broadcast_dims should preserve the full weight along broadcast axes."""
    M, N = 10, 5
    weight = torch.randn(N, dtype=torch.complex64)
    # broadcast_dims=("M",) means M is not indexed in the weight
    A = Diagonal(weight, ioshape=("M", "N"), broadcast_dims=["M"])
    # Split along the non-broadcast dim N
    ibatch = [slice(None), slice(0, 3)]
    obatch = [slice(None), slice(0, 3)]
    A_split = A.split_forward(ibatch, obatch)
    assert A_split.weight.shape == (3,)
    # Split along the broadcast dim M — weight should stay the same
    ibatch_m = [slice(0, 4), slice(None)]
    obatch_m = [slice(0, 4), slice(None)]
    A_split_m = A.split_forward(ibatch_m, obatch_m)
    assert torch.allclose(A_split_m.weight, weight)
