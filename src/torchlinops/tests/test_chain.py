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


def test_chain_shape_mismatch_raises():
    """Chain should raise ValueError if consecutive shapes don't match."""
    A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
    B = Dense(torch.randn(5, 4), ("Q", "P"), ("P",), ("Q",))
    with pytest.raises(ValueError, match="Mismatched shape"):
        Chain(A, B)


def test_chain_size_conflicting_raises():
    """Chain.size() should raise ValueError when sub-linops report conflicting sizes."""
    import torch.nn as nn
    from torchlinops.linops.chain import Chain as Ch
    from torchlinops.nameddim import NamedShape as NS

    # Build two Dense linops that both claim to know dim "M" but disagree on its size
    A = Dense(torch.randn(4, 4), ("M", "M"), ("M",), ("M",))
    B = Dense(torch.randn(4, 4), ("M", "M"), ("M",), ("M",))
    C = Chain(A, B)

    # Monkey-patch size() on the sub-linops to report conflicting values
    C[0].size = lambda dim: 4 if str(dim) == "M" else None
    C[1].size = lambda dim: 5 if str(dim) == "M" else None

    with pytest.raises(ValueError, match="Conflicting"):
        C.size("M")


class TestSharedLinopCopying:
    """Tests for automatic shallow copying of shared linops in Chain."""

    def test_chain_shared_linop_copies(self):
        """Shared linop in Chain should be copied."""
        A = Dense(torch.randn(4, 4), ("M", "M"), ("M",), ("M",))
        chain = Chain(A, A)

        assert chain.linops[0] is not chain.linops[1]

    def test_chain_shared_linop_shallow_copy(self):
        """Shallow copy should share tensor storage."""
        A = Dense(torch.randn(4, 4), ("M", "M"), ("M",), ("M",))
        chain = Chain(A, A)

        assert chain.linops[0].weight.data_ptr() == chain.linops[1].weight.data_ptr()

    def test_chain_multiple_shared_copies(self):
        """Multiple copies of same linop should all be independent objects."""
        A = Dense(torch.randn(4, 4), ("M", "M"), ("M",), ("M",))
        chain = Chain(A, A, A)

        assert chain.linops[0] is not chain.linops[1]
        assert chain.linops[1] is not chain.linops[2]
        assert chain.linops[0] is not chain.linops[2]

        assert chain.linops[0].weight.data_ptr() == chain.linops[1].weight.data_ptr()
        assert chain.linops[1].weight.data_ptr() == chain.linops[2].weight.data_ptr()

    def test_chain_execution_after_copy(self):
        """Chain should execute correctly after copying shared linops."""
        A = Dense(torch.randn(4, 4), ("M", "M"), ("M",), ("M",))
        chain = Chain(A, A)

        x = torch.randn(4)
        y = chain(x)
        assert y.shape == (4,)

    def test_chain_input_listener_independent(self):
        """Copied linops in Chain should have independent input_listener references."""
        A = Dense(torch.randn(4, 4), ("M", "M"), ("M",), ("M",))
        chain = Chain(A, A)

        first = chain.linops[0]
        second = chain.linops[1]
        assert first._input_listener._obj == chain
        assert first._input_listener._attr == "input_listener"
        assert id(second._input_listener._obj) != id(first._input_listener._obj)
        assert second._input_listener._attr != first._input_listener._attr
        assert chain.linops[0] is not chain.linops[1]
