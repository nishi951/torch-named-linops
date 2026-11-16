import pytest

from torchlinops.nameddim import unify


def test_unify_basic():
    assert unify([("A", "B"), ("A1", "B1")]) == ("A1", "B1")
    assert unify([("()", "B"), ("A", "B")]) == ("A", "B")
    assert unify([("()", "B"), ("A", "()")]) == ("A", "B")


def test_unify_ellipses():
    assert unify([("...", "C", "D"), ("A", "B", "C", "D")]) == ("A", "B", "C", "D")
    assert unify([("...", "C", "D"), ("A", "B", "()", "()")]) == ("A", "B", "C", "D")
    assert unify([("...", "C", "D"), ("...", "B", "()", "()")]) == (
        "...",
        "B",
        "C",
        "D",
    )


def test_unify_incompatible_raises():
    """unify should raise ValueError for incompatible shapes."""
    with pytest.raises(ValueError):
        unify([("A", "B"), ("C",)])
