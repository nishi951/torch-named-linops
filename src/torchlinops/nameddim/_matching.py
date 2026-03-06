from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional, Sequence, Tuple
from warnings import warn

from ._nameddim import ANY, ELLIPSES, NamedDimension as ND

__all__ = ["partition", "isequal", "iscompatible", "max_shape", "standardize_shapes"]


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


def isequal(
    shape1: Sequence,
    shape2: Sequence,
    return_assignments: bool = False,
) -> bool | tuple[bool, Optional[dict[int, list]]]:
    """Test if two sequences with ellipses are length-compatible and value-compatible.

    Implemented with bottom-up DP

    Parameters
    ----------
    shape1, shape2 : Sequence
        The sequences of tokens to compare.
    ELLIPSES : str, default = "..."
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
    >>> isequal(("A", "B", "C"), ("A", "...", "C"))
    True
    >>> isequal(("...", "A", "C", "..."), ("...",))
    True
    >>> isequal(("...", "A", "C"), ("B", "C"))
    False

    # Wildcards
    >>> isequal(("A", "B"), ("A", "()"))
    True
    >>> isequal(("A",), ("()", "()"))
    False

    # Think about this one...
    >>> isequal(("...", "A", "C", "..."), ("...", "A"))
    True
    """
    ptrs = [[(0, 0) for _ in range(len(shape2) + 1)] for _ in range(len(shape1) + 1)]
    # Base cases
    ptrs[0][0] = (0, 0)  # True (note that bool(tuple()) == False)
    for i in range(1, len(shape1) + 1):
        ptrs[i][0] = (-1, 0) if shape1[0] == ELLIPSES else None
    for j in range(1, len(shape2) + 1):
        ptrs[0][j] = (0, -1) if shape2[0] == ELLIPSES else None
    for i in range(1, len(shape1) + 1):
        for j in range(1, len(shape2) + 1):
            if ptrs[i - 1][j - 1]:
                if shape1[i - 1] == ELLIPSES or shape2[j - 1] == ELLIPSES:
                    val = (-1, -1)
                elif shape1[i - 1] == shape2[j - 1]:
                    val = (-1, -1)
                elif shape1[i - 1] == ANY or shape2[j - 1] == ANY:
                    val = (-1, -1)
                else:
                    val = None
            elif ptrs[i - 1][j]:
                if shape2[j - 1] == ELLIPSES:
                    val = (-1, 0)
                else:
                    val = None
            elif ptrs[i][j - 1]:
                if shape1[i - 1] == ELLIPSES:
                    val = (0, -1)
                else:
                    val = None
            else:
                val = None
            ptrs[i][j] = val

    if return_assignments:
        if not ptrs[-1][-1]:
            return False, None
        # Traverse in reverse order
        assignments = defaultdict(list)
        row, col = len(shape1), len(shape2)
        while row > 0 and col > 0:
            assignments[row - 1].append(col - 1)
            drow, dcol = ptrs[row][col]
            row = row + drow
            col = col + dcol
        return True, assignments
    return bool(ptrs[-1][-1])


# def get_assignments(ptrs):
#     """Assign indices from first to second sequence."""
#     len_shape1 = len(ptrs[0]) - 1
#     len_shape2 = len(ptrs) - 1
#     assignment = defaultdict(list)
#     row = 1
#     for col in range(1, len_shape1 + 1):
#         while row < len_shape2 + 1 and ptrs[row][col]:
#             assignment[col - 1].append(row - 1)
#             row += 1
#     return assignment


def iscompatible(
    shape1: Sequence,
    shape2: Sequence,
    return_assignments: bool = False,
) -> bool | tuple[bool, Optional[dict]]:
    """Whether the two shapes are length-compatible.

    >>> iscompatible(("A","B"), ("C", "D"))
    True
    >>> iscompatible(("...",), ("A","B"))
    True
    >>> iscompatible(("B","..."), ("A","..."))
    True
    >>> iscompatible(("...",), tuple())
    True

    """
    if isinstance(shape1, str) or isinstance(shape2, str):
        warn(
            f"Strings detected in iscompatible call - are you sure this is what you want? {shape1}, {shape2}"
        )

    shape1 = [ANY if s != ELLIPSES else ELLIPSES for s in deepcopy(shape1)]
    shape2 = [ANY if s != ELLIPSES else ELLIPSES for s in deepcopy(shape2)]
    return isequal(shape1, shape2, return_assignments=return_assignments)


def max_shape(shapes):
    """Find the highest-valued shape according to some heuristics.

    Examples
    --------
    >>> max_shape([("N","Q"), ("N", "Q1")])
    (N, Q1)
    >>> max_shape([("A","B"), ("A1", "B1")])
    (A1, B1)
    >>> max_shape([("...",), ("A","B")])
    (A, B)
    >>> max_shape([("A","..."), ("A2","...")])
    (A2, ...)
    """
    # Normalize inputs
    shapes = [tuple(ND.infer(s) for s in shape) for shape in shapes]
    max_shape = shapes[0]
    max_shape_names_only = [ND(d.name) for d in max_shape]
    for shape in shapes[1:]:
        shape_names_only = [ND(d.name) for d in shape]
        iscompat, assignments = iscompatible(
            max_shape_names_only, shape_names_only, return_assignments=True
        )
        if not iscompat:
            raise ValueError(
                f"Shape incompatibilty detected in concat: {max_shape} not compatible with {shape}"
            )
        new_max_shape = []
        for i in range(len(max_shape)):
            orig_dim = max_shape[i]
            # Sort to preserve original order
            new_dims = [shape[j] for j in sorted(assignments[i])]
            # Heuristic
            # 1. Ellipses and Any lose to everything
            if orig_dim == ELLIPSES or orig_dim == ANY:
                new_max_shape.extend(new_dims)
            elif ELLIPSES in new_dims or ANY in new_dims:
                new_max_shape.append(orig_dim)
            else:
                assert len(new_dims) == 1, (
                    f"Expected singleton assignment dim but got {new_dims}"
                )
                # 2. Dim vs. dim rules
                #   a. If the dim's "letter" is the same, pick the one with larger number
                new_dim = new_dims[0]
                assert new_dim.name == orig_dim.name, (
                    f"Expected singleton assignment to have matching letter but got letters {new_dim} != {orig_dim}"
                )
                new_max_shape.append(orig_dim if orig_dim.i >= new_dim.i else new_dim)
        max_shape = new_max_shape
    return tuple(max_shape)


def standardize_shapes(linops, shape):
    for linop in linops:
        linop.ishape = shape.ishape
        linop.oshape = shape.oshape
    return linops


if __name__ == "__main__":
    import doctest

    doctest.testmod()
