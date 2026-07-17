import torch
import pytest
from torchlinops import SimpleLinop


def test_simple_linop_forward():
    """Test forward operation with plain callable."""
    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 2 * y,
    )
    x = torch.randn(10, 20)
    y = A(x)
    expected = 2 * x
    assert torch.allclose(y, expected)


def test_simple_linop_adjoint_call():
    """Test adjoint operation via .H."""
    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 3 * y,
    )
    y = torch.randn(10, 20)
    z = A.H(y)
    expected = 3 * y
    assert torch.allclose(z, expected)


def test_simple_linop_normal_default():
    """Test normal operation with default (adj_fn(fn(x)))."""
    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 3 * y,
    )
    x = torch.randn(10, 20)
    z = A.N(x)
    # Normal should be adj_fn(fn(x)) = 3 * (2 * x) = 6 * x
    expected = 6 * x
    assert torch.allclose(z, expected)


def test_simple_linop_normal_custom():
    """Test normal operation with custom callable."""
    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 3 * y,
        normal=lambda x: 10 * x,
    )
    x = torch.randn(10, 20)
    z = A.N(x)
    expected = 10 * x
    assert torch.allclose(z, expected)
