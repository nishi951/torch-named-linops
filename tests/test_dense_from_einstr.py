import pytest
import torch

from torchlinops import Dense


def test_dense_from_einstr_basic():
    """Dense.from_einstr should parse einsum-style strings correctly."""
    W = torch.randn(3, 7)
    A = Dense.from_einstr(W, "MN,N->M")
    
    assert A.weightshape == ("M", "N")
    assert A.ishape == ("N",)
    assert A.oshape == ("M",)


def test_dense_from_einstr_forward_correctness():
    """Dense.from_einstr should produce correct forward pass results."""
    W = torch.randn(4, 3, dtype=torch.complex64)
    A = Dense.from_einstr(W, "MK,K->M")
    
    # Compare with equivalent Dense construction
    A_ref = Dense(W, ("M", "K"), ("K",), ("M",))
    
    x = torch.randn(3, dtype=torch.complex64)
    assert torch.allclose(A(x), A_ref(x), rtol=1e-5, atol=1e-5)


def test_dense_from_einstr_broadcast_dims():
    """Dense.from_einstr should handle broadcast_dims correctly."""
    W = torch.randn(1, 4, 3)  # first dim is broadcast
    A = Dense.from_einstr(W, "BMK,BK->BM", broadcast_dims=["B"])
    
    assert A.weightshape == ("B", "M", "K")
    assert A.broadcast_dims == ("B",)  # Stored as tuple
    
    # Invalid broadcast_dim should raise
    with pytest.raises(ValueError, match="broadcast_dim .* not in weightshape"):
        Dense.from_einstr(W, "BMK,BK->BM", broadcast_dims=["X"])
