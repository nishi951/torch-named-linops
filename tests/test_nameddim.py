import copy

import pytest

from torchlinops import ND, NamedDimension
from torchlinops.nameddim import is_any, is_ellipses, coalesce


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

    # Any and Ellipses
    assert ND.infer("(321)").i == 321
    assert ND.infer("...453").i == 453


def test_nameddim_add():
    K = NamedDimension("K")
    K4 = K + 4
    assert K4.i == 4, K4.name == "K"


def test_nameddim_ellipsis():
    ellipsis = ND("...")
    assert ellipsis == "..."
    assert ellipsis == ND("...")


def test_is_any():
    assert is_any("()")
    assert is_any("(123)")
    assert is_any(ND("()"))
    assert is_any(ND.infer("(123)"))


def test_is_ellipses():
    assert is_ellipses("...")
    assert is_ellipses("...321")
    assert is_ellipses(ND("..."))
    assert is_ellipses(ND.infer("...321"))


def test_nameddim_hash_lookup():
    foo = {"P": 1}
    assert foo[ND("P")] == 1

    bar = {ND("Q"): 2}
    assert bar["Q"] == 2


def test_ellipses_coalesce():
    assert tuple(coalesce(("...", "..."))) == ("...",)
    assert tuple(coalesce(("...", "...1"))) == ("...", "...1")
    assert tuple(coalesce(("...", "...", "...", "...1"))) == ("...", "...1")


def test_nameddim_infer_list():
    """ND.infer on a list should return a list of ND objects."""
    result = ND.infer(["A", "B", "C"])
    assert isinstance(result, list)
    assert all(isinstance(d, ND) for d in result)
    assert str(result[0]) == "A"
    assert str(result[1]) == "B"


def test_nameddim_infer_tuple():
    """ND.infer on a tuple should return a tuple of ND objects."""
    result = ND.infer(("Nx", "Ny"))
    assert isinstance(result, tuple)
    assert str(result[0]) == "Nx"


def test_namedshape_add_incompatible_raises():
    """NamedShape + object with no ishape/oshape should raise an error."""
    from torchlinops.nameddim import NamedShape as NS

    s = NS(("A",), ("B",))
    # The __add__ impl catches TypeError from tuple + non-tuple and re-raises TypeError.
    # When right has no ishape attribute at all it raises AttributeError first.
    with pytest.raises((TypeError, AttributeError)):
        _ = s + object()
