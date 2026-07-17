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


def test_simple_linop_custom_shapes():
    """Test custom ishape and oshape."""
    A = SimpleLinop(
        forward=lambda x: x,
        adjoint=lambda y: y,
        ishape=("Nx", "Ny"),
        oshape=("Mx", "My"),
    )
    assert A.ishape == ("Nx", "Ny")
    assert A.oshape == ("Mx", "My")


def test_simple_linop_default_shapes():
    """Test default shapes are (...,)."""
    A = SimpleLinop(
        forward=lambda x: x,
        adjoint=lambda y: y,
    )
    assert A.ishape == ("...",)
    assert A.oshape == ("...",)


def test_simple_linop_oshape_defaults_to_ishape():
    """Test that oshape defaults to ishape when not provided."""
    A = SimpleLinop(
        forward=lambda x: x,
        adjoint=lambda y: y,
        ishape=("Nx", "Ny"),
    )
    assert A.ishape == ("Nx", "Ny")
    assert A.oshape == ("Nx", "Ny")


def test_simple_linop_name():
    """Test custom name."""
    A = SimpleLinop(
        forward=lambda x: x,
        adjoint=lambda y: y,
        name="MyLinop",
    )
    assert A.name == "MyLinop"
    assert "MyLinop" in repr(A)


def test_simple_linop_adjoint_name():
    """Test that adjoint gets .H suffix in name."""
    A = SimpleLinop(
        forward=lambda x: x,
        adjoint=lambda y: y,
        name="MyLinop",
    )
    A_H = A.H
    assert A_H.name == "MyLinop.H"
    assert "MyLinop.H" in repr(A_H)


def test_simple_linop_no_name():
    """Test that linop without name uses class name."""
    A = SimpleLinop(
        forward=lambda x: x,
        adjoint=lambda y: y,
    )
    assert A.name == "SimpleLinop"
    assert "SimpleLinop" in repr(A)


def test_simple_linop_in_chain():
    """Test SimpleLinop in Chain container."""
    from torchlinops import Chain, Identity

    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 2 * y,
        ishape=("N",),
    )
    B = Identity(ishape=("N",))
    C = Chain(A, B)

    x = torch.randn(10)
    y = C(x)
    expected = 2 * x
    assert torch.allclose(y, expected)


def test_simple_linop_in_add():
    """Test SimpleLinop in Add container."""
    from torchlinops import Add

    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 2 * y,
    )
    B = SimpleLinop(
        forward=lambda x: 3 * x,
        adjoint=lambda y: 3 * y,
    )
    C = Add(A, B)

    x = torch.randn(10, 20)
    y = C(x)
    expected = 5 * x
    assert torch.allclose(y, expected)


def test_simple_linop_in_stack():
    """Test SimpleLinop in Stack container."""
    from torchlinops import Stack

    A = SimpleLinop(
        forward=lambda x: 2 * x,
        adjoint=lambda y: 2 * y,
        ishape=("N",),
        oshape=("N",),
    )
    B = SimpleLinop(
        forward=lambda x: 3 * x,
        adjoint=lambda y: 3 * y,
        ishape=("N",),
        oshape=("N",),
    )
    C = Stack(A, B, odim_and_idx=("K", 0))

    x = torch.randn(10)
    y = C(x)
    assert y.shape[0] == 2
    assert torch.allclose(y[0], 2 * x)
    assert torch.allclose(y[1], 3 * x)


def test_simple_linop_in_concat():
    """Test SimpleLinop in Concat container."""
    from torchlinops import Concat, Dense

    A = Dense(torch.randn(10, 10), ("M", "N"), ishape=("N",), oshape=("M",))
    B = Dense(torch.randn(10, 10), ("M", "N"), ishape=("N",), oshape=("M",))
    C = Concat(A, B, odim="M")

    x = torch.randn(10)
    y = C(x)
    assert y.shape[0] == 20
    assert torch.allclose(y[:10], A(x))
    assert torch.allclose(y[10:], B(x))
