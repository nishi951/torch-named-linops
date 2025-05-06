import pytest

from torchlinops import NamedLinop, NS


def test_namedlinop_adjoint():
    A = NamedLinop(NS(("A", "B"), ("C",)))
    AH = A.H


def test_namedlinop_split():
    A = NamedLinop(NS(("A", "B"), ("C",)))
    A1 = A.split(A, [slice(None), slice(None)], [slice(None)])


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
