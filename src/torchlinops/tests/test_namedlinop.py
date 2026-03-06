import copy

import torch

from torchlinops import NS, NamedLinop, Dense, Diagonal, Identity


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


def test_normal_of_normal():
    """A.N.N(x) should equal A.H(A(A.H(A(x)))) for a Diagonal."""
    weight = torch.randn(5, dtype=torch.complex64)
    A = Diagonal(weight, ("N",))
    x = torch.randn(5, dtype=torch.complex64)
    nn_x = A.N.N(x)
    expected = A.H(A(A.H(A(x))))
    assert torch.allclose(nn_x, expected, rtol=1e-5)


def test_shallow_copy_resets_adjoint_and_shares_params():
    """copy.copy(A) should share parameters but reset _adjoint/_normal and create new _shape."""
    N = 5
    weight = torch.randn(N, dtype=torch.complex64)
    A = Diagonal(weight, ("N",))
    # Prime the cache
    _ = A.H

    B = copy.copy(A)
    # _shape is a new object
    assert B._shape is not A._shape
    # Parameter tensor is shared (same storage)
    assert B.weight.data_ptr() == A.weight.data_ptr()
    # Adjoint cache is cleared on copy
    assert B._adjoint is None


def test_double_adjoint_suffix_logic():
    """_update_suffix should toggle '.H' on each adjoint call."""
    weight = torch.randn(4, 3, dtype=torch.complex64)
    A = Dense(weight, ("M", "N"), ("N",), ("M",))
    # Manually call _update_suffix to verify toggle logic
    A._update_suffix(adjoint=True)
    assert A._suffix == ".H"
    A._update_suffix(adjoint=True)  # second call strips it
    assert A._suffix == ""


def test_flatten_base_case():
    """NamedLinop.flatten() should return [self] for a non-chain linop."""
    A = Identity(("N",))
    assert A.flatten() == [A]
