import pytest
import torch

from torchlinops import Truncate, PadDim
from torchlinops.tests.test_base import BaseNamedLinopTests


class TestTruncate(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        ishape = ("N", "M")
        oshape = ("N", "M")
        return (
            Truncate(dim=1, from_length=5, to_length=3, ishape=ishape, oshape=oshape),
            torch.randn(4, 5, dtype=torch.complex64),
            torch.randn(4, 3, dtype=torch.complex64),
        )

    def test_forward_shape(self, linop_input_output):
        A, x, y = linop_input_output
        result = A(x)
        assert result.shape == (4, 3)

    def test_adjoint_returns_paddim(self, linop_input_output):
        A, _, _ = linop_input_output
        adj = A.adjoint()
        assert isinstance(adj, PadDim)
        assert adj.dim == A.dim
        assert adj.from_length == A.to_length
        assert adj.to_length == A.from_length

    def test_split_error(self, linop_input_output):
        A, _, _ = linop_input_output
        with pytest.raises(ValueError, match="Cannot slice"):
            A.split(A, {"M": slice(0, 3)})

    def test_split_valid(self, linop_input_output):
        A, _, _ = linop_input_output
        result = A.split(A, {"N": slice(0, 2)})
        assert isinstance(result, Truncate)
        assert result.dim == 1
        assert result.from_length == 5
        assert result.to_length == 3


class TestPadDim(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        ishape = ("N", "M")
        oshape = ("N", "M")
        return (
            PadDim(dim=1, from_length=3, to_length=5, ishape=ishape, oshape=oshape),
            torch.randn(4, 3, dtype=torch.complex64),
            torch.randn(4, 5, dtype=torch.complex64),
        )

    def test_forward_shape(self, linop_input_output):
        A, x, y = linop_input_output
        result = A(x)
        assert result.shape == (4, 5)

    def test_adjoint_returns_truncate(self, linop_input_output):
        A, _, _ = linop_input_output
        adj = A.adjoint()
        assert isinstance(adj, Truncate)
        assert adj.dim == A.dim
        assert adj.from_length == A.to_length
        assert adj.to_length == A.from_length

    def test_split_error(self, linop_input_output):
        A, _, _ = linop_input_output
        with pytest.raises(ValueError, match="Cannot slice"):
            A.split(A, {"M": slice(0, 3)})

    def test_split_valid(self, linop_input_output):
        A, _, _ = linop_input_output
        result = A.split(A, {"N": slice(0, 2)})
        assert isinstance(result, PadDim)
        assert result.dim == 1
        assert result.from_length == 3
        assert result.to_length == 5


# --- Validation error tests ---


def test_truncate_negative_from_length():
    with pytest.raises(ValueError):
        Truncate(dim=0, from_length=-1, to_length=3, ishape=("N",), oshape=("N",))


def test_truncate_invalid_to_length():
    with pytest.raises(ValueError):
        Truncate(dim=0, from_length=5, to_length=6, ishape=("N",), oshape=("N",))


def test_truncate_fn_size_mismatch():
    T = Truncate(dim=0, from_length=5, to_length=3, ishape=("N",), oshape=("N",))
    with pytest.raises(ValueError):
        T(torch.randn(10))


def test_truncate_adj_fn_size_mismatch():
    T = Truncate(dim=0, from_length=5, to_length=3, ishape=("N",), oshape=("N",))
    with pytest.raises(ValueError):
        T.H(torch.randn(10))


def test_truncate_normal_fn_size_mismatch():
    T = Truncate(dim=0, from_length=5, to_length=3, ishape=("N",), oshape=("N",))
    with pytest.raises(ValueError):
        T.N(torch.randn(10))


def test_paddim_invalid_to_length():
    with pytest.raises(ValueError):
        PadDim(dim=0, from_length=3, to_length=-1, ishape=("N",), oshape=("N",))


def test_paddim_invalid_from_length():
    with pytest.raises(ValueError):
        PadDim(dim=0, from_length=6, to_length=5, ishape=("N",), oshape=("N",))


def test_paddim_fn_size_mismatch():
    P = PadDim(dim=0, from_length=3, to_length=5, ishape=("N",), oshape=("N",))
    with pytest.raises(ValueError):
        P(torch.randn(10))


def test_paddim_adj_fn_size_mismatch():
    P = PadDim(dim=0, from_length=3, to_length=5, ishape=("N",), oshape=("N",))
    with pytest.raises(ValueError):
        P.H(torch.randn(10))


def test_paddim_adjoint_valid_input():
    """PadDim.adj_fn previously had a NameError (truncate instead of padend).
    This test ensures the adjoint executes correctly on a valid input."""
    P = PadDim(dim=0, from_length=3, to_length=5, ishape=("N",), oshape=("N",))
    y = torch.randn(5, dtype=torch.complex64)
    result = P.H(y)
    assert result.shape == (3,)
    assert torch.allclose(result, y[:3])


def test_truncate_normal_with_inner():
    """Truncate.normal(inner) should compose post @ inner @ pre correctly."""
    from torchlinops import Identity

    T = Truncate(dim=0, from_length=5, to_length=3, ishape=("N",), oshape=("N",))
    inner = Identity(("N",))
    normal = T.normal(inner=inner)
    x = torch.randn(5, dtype=torch.complex64)
    expected = T.H(inner(T(x)))
    assert torch.allclose(normal(x), expected, rtol=1e-5)


def test_paddim_normal_with_inner():
    """PadDim.normal(inner) should compose post @ inner @ pre correctly."""
    from torchlinops import Identity

    P = PadDim(dim=0, from_length=3, to_length=5, ishape=("N",), oshape=("N",))
    inner = Identity(("N",))
    normal = P.normal(inner=inner)
    x = torch.randn(3, dtype=torch.complex64)
    expected = P.H(inner(P(x)))
    assert torch.allclose(normal(x), expected, rtol=1e-5)
