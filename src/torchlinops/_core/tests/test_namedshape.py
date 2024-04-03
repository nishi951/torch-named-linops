import pytest

from torchlinops import NamedShape, NamedDiagShape, NamedComboShape


def test_adjoint():
    shape = NamedShape(("A", "B"), ("C",))
    adj_shape = shape.H
    adj_shape.ishape = ("D",)
    assert shape.oshape == ("D",)


def test_normal():
    shape = NamedShape(("A", "B"), ("C",))
    norm_shape = shape.N

    norm_shape.ishape = ("D", "E")
    assert shape.ishape == ("D", "E")
    assert shape.oshape == ("C",)

    shape.ishape = ("F", "G")
    assert norm_shape.ishape == ("F", "G")
    assert norm_shape.oshape == ("F", "G")


def test_adjoint_and_normal():
    shape = NamedShape(("M", "N"), ("J", "K"))
    adj_shape = shape.H
    norm_shape = shape.N

    norm_shape.ishape = ("O", "P")
    assert adj_shape.ishape == ("J", "K")
    assert adj_shape.oshape == ("O", "P")
    assert shape.ishape == ("O", "P")
    assert shape.oshape == ("J", "K")

    adj_shape.oshape = ("Q", "R")
    assert shape.ishape == ("Q", "R")
    assert shape.oshape == ("J", "K")
    assert norm_shape.ishape == ("Q", "R")
    assert norm_shape.oshape == ("Q", "R")


def test_diag():
    shape = NamedDiagShape(("R", "S"))
    adj_shape = shape.H
    norm_shape = shape.H
    adj_shape.ishape = ("P", "Q")
    assert shape.ishape == ("P", "Q")
    assert shape.oshape == ("P", "Q")
    assert norm_shape.ishape == ("P", "Q")
    assert norm_shape.oshape == ("P", "Q")


def test_combo():
    shape = NamedComboShape(("A", "C"), ("Nx", "Ny"), ("T", "R", "K"))
    adj_shape = shape.H
    adj_shape.ishape = ("A1", "C", "T", "R", "K")

    assert shape.ishape == ("A1", "C", "Nx", "Ny")
    assert shape.oshape == ("A1", "C", "T", "R", "K")
