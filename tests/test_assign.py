from torchlinops.nameddim import AnyDim, match, Assignment


def test_assign_basic():
    assignment = match((AnyDim(), AnyDim(1)), ("A", "B"))
    output = assignment.assign((AnyDim(), AnyDim(1)))
    assert output == ("A", "B")
