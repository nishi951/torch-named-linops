import pytest
import torch

from torchlinops import Dense, Dim
from torchlinops.linops.chain import Chain
from torchlinops.tests.test_base import BaseNamedLinopTests


class TestChain(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        M, K, N = 5, 4, 3
        A = Dense(
            torch.randn(M, K, dtype=torch.complex64),
            ("M", "K"),
            ("K",),
            ("M",),
        )
        B = Dense(
            torch.randn(K, N, dtype=torch.complex64),
            ("K", "N"),
            ("N",),
            ("K",),
        )
        # B: N->K, A: K->M. Chain(B, A) -> A(B(x))
        C = Chain(B, A)
        x = torch.randn(N, dtype=torch.complex64)
        y = torch.randn(M, dtype=torch.complex64)
        return C, x, y

    def test_normal_level2(self, linop_input_output):
        pass


def test_chain_getitem():
    A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
    B = Dense(torch.randn(3, 2), ("K", "N"), ("N",), ("K",))
    C = Chain(B, A)
    assert len(C) == 2
    assert C[0] is B
    sub = C[0:1]
    assert isinstance(sub, Chain)


def test_chain_repr():
    A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
    B = Dense(torch.randn(3, 2), ("K", "N"), ("N",), ("K",))
    C = Chain(B, A)
    r = repr(C)
    assert "Chain" in r or "Dense" in r


def test_chain_dims():
    A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
    B = Dense(torch.randn(3, 2), ("K", "N"), ("N",), ("K",))
    C = Chain(B, A)
    dims = C.dims
    assert "M" in dims
    assert "N" in dims
    assert "K" in dims
