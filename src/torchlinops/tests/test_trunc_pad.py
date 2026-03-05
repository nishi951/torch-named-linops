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
