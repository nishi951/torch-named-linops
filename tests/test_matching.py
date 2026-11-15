from torchlinops.nameddim.utils import match


def test_match_basic():
    assignment = match(("A", "B"), ("A", "B"))
    assert assignment is not None
    assert assignment.mapping == (0, 1)


def test_match_ellipses_on_left():
    assignment = match(("...", "C", "D"), ("A", "B", "C", "D"))
    assert assignment is not None
    assert assignment.mapping == ((0, 1), 2, 3)


def test_match_ellipses_on_right():
    assignment = match(("A",), ("...", "A"))
    assert assignment is not None
    assert assignment.mapping == (1,)


def test_match_empty_target():
    assignment = match(("A", "C"), ("...",))
    assert assignment is not None
    assert assignment.mapping == (0, 0)
