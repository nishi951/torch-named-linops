from torchlinops.nameddim import AnyDim, match, assign


def test_assign_basic():
    assignment = match((AnyDim(), AnyDim(1)), ("A", "B"))
    output = assign(assignment, (AnyDim(), AnyDim(1)))
    assert output == ("A", "B")
