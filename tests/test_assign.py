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
