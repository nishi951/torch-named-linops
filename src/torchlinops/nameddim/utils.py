"""Temporary staging area for new functionality

Once functionality is established, we will refactor the entire module.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._nameddim import ANY, ELLIPSES, shape
from ._nameddim import NamedDimension as ND

__all__ = [
    "AnyShape",
    "Assignment",
    "EmptyShape",
    "Shape",
    "match",
    "normalize",
]

Shape = tuple[ND, ...]
AnyShape = (ND.infer(ELLIPSES),)
EmptyShape = ()


def is_any(dim):
    """Check if dim is ANY or ordinal ANY."""
    if isinstance(dim, str):
        # Check if it's a string representation of ANY or ordinal ANY
        return dim == ANY or (
            dim.startswith("(") and dim.endswith(")") and dim[1:-1].isdigit()
        )
    return hasattr(dim, "name") and dim.name == ANY


def coalesce(lst, cond):
    result = []
    for x in lst:
        if cond(x) and result and cond(result[-1]):
            continue
        result.append(x)
    return result


def normalize(value: str | Sequence | None) -> Shape:
    """Canonicalize an input into a shape object.

    Coalesces adjacent ellipses together.

    Examples
    --------
    >>> normalize("AB")
    (A, B)
    >>> normalize(("A", "...", "...", "B"))
    (A, ..., B)
    >>> normalize(())
    ()
    >>> normalize(None)
    (...,)
    >>> normalize("(1)")
    ((1),)
    """
    if value is None:
        return AnyShape
    if isinstance(value, list | tuple):
        output = [ND.infer(s) for s in value]
    elif isinstance(value, str):
        output = [ND.infer(s) for s in shape(value)]
    # Coalesce ellipses
    output = coalesce(output, lambda x: x == ELLIPSES)
    return tuple(output)


@dataclass(slots=True, frozen=True)
class Assignment:
    source: tuple[Any, ...]
    """The input from which to assign."""
    target: tuple[Any, ...]
    """The output to assign to."""
    mapping: tuple[tuple[int, ...] | int, ...]
    """A mapping from source position to some number of target positions"""

    def __repr__(self):
        return f"{type(self).__name__}({self.source} -> {self.target})"

    def apply(self, dim: ND):
        src = self.source.index(dim)
        idx = self.mapping[src]
        if isinstance(idx, tuple):
            return tuple(self.target[i] for i in idx)
        return self.target[idx]

    def assign(self, source: Shape):
        output = []
        for src in source:
            if src == ELLIPSES:
                output.extend(self.apply(src))
            else:
                output.append(self.apply(src))
        return normalize(output)


def match(source: tuple, target: tuple) -> Assignment | None:
    """

    Examples
    --------
    >>> match(("A", "B"), ("A", "B")).mapping
    (0, 1)
    >>> match(("A", "C"), ("A",)) # None
    >>> match(("A", "C"), tuple()) # None
    >>> match(("A", "C"), ("...",)).mapping
    (0, 0)
    >>> match(("A", "C", "..."), ("...",)).mapping
    (0, 0, (0,))
    >>> match(("A", "B", "C"), ("A", "...", "C")).mapping
    (0, 1, 2)
    >>> match(("...", "C", "D"), ("A", "B", "C", "D")).mapping
    ((0, 1), 2, 3)

    # Wildcards
    >>> match(("A", "B"), ("A", "()")).mapping
    (0, 1)
    >>> match(("A",), ("()", "()"))

    # Think about this one...
    >>> match(("...", "A", "C", "..."), ("...", "A")).mapping
    ((0,), 0, 0, (1,))

    >>> match(("A",), ("...", "A")).mapping
    (1,)

    """
    # Rows = source
    # Cols = target
    ptrs: list[list] = [
        [None for _ in range(len(target) + 1)] for _ in range(len(source) + 1)
    ]
    # Base cases
    ptrs[0][0] = (0, 0)  # True (note that bool(tuple()) == False)
    for i in range(1, len(source) + 1):
        ptrs[i][0] = (-1, 0) if source[0] == ELLIPSES else None
    for j in range(1, len(target) + 1):
        ptrs[0][j] = (0, -1) if target[0] == ELLIPSES else None

    # Dynamic programming
    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            val = None
            if ptrs[i - 1][j - 1] and (
                source[i - 1] == ELLIPSES
                or target[j - 1] == ELLIPSES
                or source[i - 1] == target[j - 1]
                or is_any(source[i - 1])
                or is_any(target[j - 1])
            ):
                val = (-1, -1)
            if val is None and ptrs[i - 1][j] and target[j - 1] == ELLIPSES:
                val = (-1, 0)
            if val is None and ptrs[i][j - 1] and source[i - 1] == ELLIPSES:
                val = (0, -1)
            ptrs[i][j] = val

    if not ptrs[-1][-1]:
        return None

    # Unpack the DP to get the assignments
    mapping = [[] for _ in range(len(source))]
    row, col = len(source), len(target)
    while True:
        if row == 0 and col == 0:
            break
        assert ptrs[row][col] is not None, "Detected None ptr in passing match."
        mapping[max(0, row - 1)].insert(0, max(0, col - 1))
        drow, dcol = ptrs[row][col]
        row = row + drow
        col = col + dcol

    # Assign non-ELLIPSES strictly to non-ELLIPSES
    output = []
    for src, lst in zip(source, mapping):
        if src == ELLIPSES:
            output.append(tuple(lst))
        else:
            # Only assign to ELLIPSES if its the only option
            if all(target[elem] == ELLIPSES for elem in lst):
                output.append(lst[0])
            else:
                # Assign to the non-ELLIPSES element
                for elem in lst:
                    if target[elem] != ELLIPSES:
                        output.append(elem)
                        break
    mapping = tuple(output)
    return Assignment(source, target, mapping)


def unify(): ...
