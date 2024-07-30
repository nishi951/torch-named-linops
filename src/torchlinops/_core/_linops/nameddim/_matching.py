from typing import Sequence, Any, Tuple

__all__ = ["partition", "isequal"]


def partition(seq: Sequence, val: Any) -> Tuple[Sequence, Sequence, Sequence]:
    """Split a sequence on the first occurence of some value

    Examples
    --------
    >>> partition(("A", "B", "C"), "B")
    (('A',), ('B',), ('C',))

    >>> partition(("A", "B", "C"), "C")
    (('A', 'B'), ('C',), ())

    >>> partition(("A", "B", "C"), "D")
    (('A', 'B', 'C'), (), ())


    """
    if val not in seq:
        return seq, type(seq)(), type(seq)()
    n = seq.index(val)
    first = seq[:n]
    middle = type(seq)((val,))
    last = seq[n + 1 :]
    return first, middle, last


def isequal(shape1: Sequence, shape2: Sequence, WILD: str = "...") -> bool:
    """Test if two sequences with ellipses are compatible

    Implemented with bottom-up DP

    Parameters
    ----------
    shape1, shape2 : Sequence
        The sequences of tokens to compare.
    WILD : str, default = "..."
        The wildcard that can match any number of tokens.

    Returns
    -------
    bool
        Whether shape1 and shape2 are compatible.

    Examples
    --------

    >>> isequal(("A", "B"), ("A", "B"))
    True
    >>> isequal(("A", "C"), ("A",))
    False
    >>> isequal(("A", "C"), tuple())
    False
    >>> isequal(("A", "C"), ("...",))
    True
    >>> isequal(("A", "C", "..."), ("...",))
    True
    >>> isequal(("...", "A", "C", "..."), ("...",))
    True
    >>> isequal(("...", "A", "C"), ("B", "C"))
    False

    # Think about this one...
    >>> isequal(("...", "A", "C", "..."), ("...", "A"))
    True
    """
    costs = [[None] * (len(shape2) + 1) for _ in range(len(shape1) + 1)]
    # Base cases
    costs[0][0] = True
    for i in range(1, len(shape1) + 1):
        costs[i][0] = shape1[0] == WILD
    for j in range(1, len(shape2) + 1):
        costs[0][j] = shape2[0] == WILD
    for i in range(1, len(shape1) + 1):
        for j in range(1, len(shape2) + 1):
            if costs[i - 1][j - 1]:
                if shape1[i - 1] == WILD or shape2[j - 1] == WILD:
                    val = True
                else:
                    val = shape1[i - 1] == shape2[j - 1]
            elif costs[i - 1][j]:
                if shape2[j - 1] == WILD:
                    val = True
                else:
                    val = shape1[i - 1] == shape2[j - 1]
            elif costs[i][j - 1]:
                if shape1[i - 1] == WILD:
                    val = True
                else:
                    val = shape1[i - 1] == shape2[j - 1]
            else:
                val = False
            costs[i][j] = val
    return costs[-1][-1]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
