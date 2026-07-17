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


def test_simple_linop_adjoint_is_simple_linop():
    """Test that .H returns a SimpleLinop instance."""
    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 3 * y,
    )
    A_H = A.H
    assert isinstance(A_H, SimpleLinop)


def test_simple_linop_adjoint_swaps_callables():
    """Test that .H swaps forward and adjoint callables."""
    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 3 * y,
    )
    A_H = A.H

    x = torch.randn(10, 20)
    # A.H forward should use A's adjoint callable
    y = A_H(x)
    expected = 3 * x
    assert torch.allclose(y, expected)

    # A.H adjoint should use A's forward callable
    z = A_H.H(x)
    expected = 2 * x
    assert torch.allclose(z, expected)


def test_simple_linop_adjoint_swaps_shapes():
    """Test that .H swaps ishape and oshape."""
    A = SimpleLinop(
        forward=lambda x: x,
        adjoint=lambda y: y,
        ishape=("Nx", "Ny"),
        oshape=("Mx", "My"),
    )
    A_H = A.H
    assert A_H.ishape == ("Mx", "My")
    assert A_H.oshape == ("Nx", "Ny")


def test_simple_linop_adjoint_normal_computed_correctly():
    """Test that adjoint's normal is computed from swapped callables."""
    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 3 * y,
        normal=lambda x: 10 * x,  # Custom normal for A
    )
    A_H = A.H

    x = torch.randn(10, 20)
    # A.H normal should be forward(adjoint(x)) = 2 * (3 * x) = 6 * x
    # NOT the custom normal from A
    z = A_H.N(x)
    expected = 6 * x
    assert torch.allclose(z, expected)


def test_simple_linop_composition():
    """Test composition with @ operator."""
    from torchlinops import Identity

    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 2 * y,
        ishape=("N",),
    )
    B = Identity(ishape=("N",))

    # Compose: B @ A should apply A first, then B
    C = B @ A
    x = torch.randn(10)
    y = C(x)
    expected = 2 * x
    assert torch.allclose(y, expected)


def test_simple_linop_addition():
    """Test addition with + operator."""
    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 2 * y,
    )
    B = SimpleLinop(
        forward=lambda x: 3 * x,
        adjoint=lambda y: 3 * y,
    )

    C = A + B
    x = torch.randn(10, 20)
    y = C(x)
    expected = 5 * x
    assert torch.allclose(y, expected)


def test_simple_linop_scalar_multiplication():
    """Test scalar multiplication with * operator."""
    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 2 * y,
    )

    B = 3 * A
    x = torch.randn(10, 20)
    y = B(x)
    expected = 6 * x
    assert torch.allclose(y, expected)
