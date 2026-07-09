import pytest
import torch

from torchlinops.functional import index, index_adjoint


def test_index_2d_bcast():
    idx = (torch.tensor([1, 0, 1])[:, None], torch.tensor([1, 2, 3])[None, :])
    vals = torch.arange(8).reshape(2, 4)
    out = vals[idx]

    out2 = index(vals, idx)
    assert (out == out2).all()


def test_index_2d_bcast_slicing():
    tidx = (torch.tensor([0, 1])[:, None], torch.tensor([1, 2, 3])[None, :])
    vals = torch.arange(8).reshape(2, 4)
    out = vals[tidx]

    idx = (slice(None), slice(1, 4))
    out2 = index(vals, idx)
    assert (out == out2).all()


def test_index_adjoint():
    idx = torch.tensor([1, 3, 2])
    idx = (idx,)
    vals = torch.tensor([[5.0, 4.0, -1.0], [5.0, 4.0, -1.0]])
    oshape = (4,)
    out = index_adjoint(vals, idx, oshape)
    assert (out[0] == torch.tensor([0.0, 5.0, -1.0, 4.0])).all()
    assert (out[0] == out[1]).all()


def test_index_nondeterministic_forward():
    """Test that index produces the same forward results as raw getitem."""
    vals = torch.randn(64, 100)
    idx = (torch.randint(0, 100, (1000,)),)

    out_ref = index(vals, idx, deterministic_backward=True)
    out_nd = index(vals, idx, deterministic_backward=False)

    assert torch.equal(out_ref, out_nd)
    assert out_ref.shape == out_nd.shape


def test_index_nondeterministic_backward():
    """Test that index_nondeterministic backward produces correct gradients."""
    vals = torch.randn(64, 100, requires_grad=True)
    idx = (torch.randint(0, 100, (1000,)),)

    # Test with index
    vals_ref = vals.detach().clone().requires_grad_(True)
    out_ref = index(vals_ref, idx, deterministic_backward=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    grad_ref = vals_ref.grad.clone()

    # Test with index, deterministic_backward=False
    vals_nd = vals.detach().clone().requires_grad_(True)
    out_nd = index(vals_nd, idx, deterministic_backward=False)
    out_nd.backward(grad_out)
    grad_nd = vals_nd.grad.clone()

    assert torch.allclose(grad_ref, grad_nd, atol=1e-6)
