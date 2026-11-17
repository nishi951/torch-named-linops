import pytest

from torchlinops.nameddim import (
    AnyDim,
    match,
    Assignment,
    skeleton,
    NamedDimension as ND,
    ELLIPSES,
)


def test_assign_basic():
    assignment = match((AnyDim(), AnyDim(1)), ("A", "B"))
    output = assignment.assign((AnyDim(), AnyDim(1)))
    assert output == ("A", "B")


def test_skeleton():
    assert skeleton(("A", "B")) == (AnyDim(0), AnyDim(1))
    assert skeleton(("A", "...", "B")) == (AnyDim(0), ND.infer(ELLIPSES), AnyDim(1))
    assert skeleton(("A", "...", "A")) == (AnyDim(0), ELLIPSES, AnyDim(0))
    assert skeleton(("A", "...", "A", "B")) == (
        AnyDim(0),
        ELLIPSES,
        AnyDim(0),
        AnyDim(1),
    )


def test_assign_then_apply_simple():
    a = match(("A", "A"), ("...", "A"))
    assert a is not None
    assert a.apply(ND.infer("A")) == "A"
    assert a.apply("B") is None


def test_assign_then_apply_hard():
    a = match(("...", "A"), ("A",))
    assert a is not None
    assert a.assign(("...", "A")) == ("A",)  # Ellipsis is deleted


def test_assign_then_apply_uniqueness():
    a = match(("()", "()"), ("...", "A"))
    assert a is not None
    assert a.apply(AnyDim()) == "A"


def test_assign_then_apply_uniqueness_failure():
    a = match(("()", "()", "()"), ("...", "A", "B"))
    with pytest.raises(ValueError, match="uniquely assign"):
        a.apply("()")


def test_assign_then_apply_ellipses():
    a = match(("...", "C", "..."), ("A", "B", "C", "A", "B"))
    assert a is not None
    assert a.apply("...") == ("A", "B")

    a = match(("...", "C", "..."), ("A", "B", "C", "D", "E"))
    assert a is not None
    with pytest.raises(ValueError, match="uniquely assign"):
        a.apply("...")

    a = match(("...", "C", "...1"), ("A", "B", "C", "D", "E"))
    assert a is not None
    assert a.apply("...") == ("A", "B")
    assert a.apply("...1") == ("D", "E")


def test_assign_then_apply_wildcard_targets():
    a = match(("...", "C", "..."), ("A", "B", "C", "..."))
    assert a is not None
    with pytest.raises(ValueError, match="uniquely assign"):
        a.apply("...")

    a = match(("A", "B"), ("()", "()"))
    assert a is not None
    assert a.apply("A") == AnyDim()
    assert a.apply("B") == AnyDim()

    a = match(("A", "B"), ("...",))
    assert a is not None
    assert a.apply("A") == "..."
    assert a.apply("B") == "..."


def test_assign_with_replace():
    a = match(("B", "N"), ("B", "N1"), replace=True)
    assert a is not None
    assert a.apply("B") == "B"
    assert a.apply("N") == "N1"

    a = match(("N",), ("...", "N1"), replace=True)
    assert a is not None
    assert a.apply("N") == "N1"
