import torch

from torchlinops import NS, NamedLinop, Dense, Diagonal


def test_namedlinop_adjoint():
    A = NamedLinop(NS(("A", "B"), ("C",)))
    AH = A.H


def test_namedlinop_split():
    A = NamedLinop(NS(("A", "B"), ("C",)))
    A1 = A.split(A, {})


def test_namedlinop_normal():
    A = NamedLinop(NS(("A", "B"), ("C",)))
    AN = A.N


def test_namedlinop_chain_normal_split():
    A = NamedLinop(NS(("A", "B"), ("C",)))
    B = NamedLinop(NS(("C",), ("C1",)))
    AB = B @ A
    ABN = AB.N


def test_namedlinop_custom_name():
    A = NamedLinop(NS(("A", "B"), ("C",)))
    assert str(A).startswith("NamedLinop")

    custom_name = "Test"
    A = NamedLinop(NS(("A", "B"), ("C",)), name=custom_name)
    assert str(A).startswith(custom_name)


def test_namedlinop_matrix_vector_mul():
    A = NamedLinop(NS(("A", "B"), ("C",)))
    x = torch.tensor(3)
    assert A(x).allclose(A @ x)


def test_linop_function_backward_broadcast():
    """LinopFunction.backward should produce grad matching A.H applied to ones."""
    M, N = 4, 3
    weight = torch.randn(M, N, dtype=torch.complex64)
    A = Dense(weight, ("M", "N"), ("N",), ("M",))
    x = torch.randn(N, dtype=torch.complex64).requires_grad_(True)
    out = A.apply(x)
    grad_out = torch.ones_like(out)
    out.real.sum().backward()
    # grad w.r.t. x should equal A.H applied to grad_out
    expected = A.H.apply(grad_out)
    assert torch.allclose(x.grad, expected, rtol=1e-4)
