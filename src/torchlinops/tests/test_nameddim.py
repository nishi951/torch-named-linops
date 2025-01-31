import copy

from torchlinops import NamedDimension, ND


def test_nameddim():
    C = NamedDimension("C")
    C1 = copy.copy(C)
    C1 = C + 1
    assert C1.i == 1

    C3 = NamedDimension.infer(C)
    assert C3.name == "C"


def test_nameddim_repr():
    A = NamedDimension("A")
    assert repr(A) == "A"
    B = NamedDimension("B", i=1)
    assert repr(B) == "B1"


def test_nameddim_infer():
    Nx = NamedDimension.infer("Nx")
    assert Nx.i == 0 and Nx.name == "Nx"
    P1 = NamedDimension.infer("P1")
    assert P1.i == 1 and P1.name == "P"
    ishape = ("C", "Nx", "Ny")
    ishape2 = NamedDimension.infer(ishape)
    assert ishape2[1] == NamedDimension("Nx")


def test_nameddim_add():
    K = NamedDimension("K")
    K4 = K + 4
    assert K4.i == 4, K4.name == "K"


def test_nameddim_ellipsis():
    ellipsis = ND("...")
    assert ellipsis == "..."
    assert ellipsis == ND("...")
