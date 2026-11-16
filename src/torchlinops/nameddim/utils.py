"""Temporary staging area for new functionality

Once functionality is established, we will refactor the entire module.
"""

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from typing import Any

from ._nameddim import ANY, ELLIPSES, AnyDim, shape
from ._nameddim import NamedDimension as ND

__all__ = [
    "ELLIPSES_MATCH",
    "AnyShape",
    "Assignment",
    "EmptyShape",
    "Shape",
    "is_any",
    "is_ellipses",
    "match",
    "normalize",
    "skeleton",
    "unify",
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


def is_ellipses(dim):
    """Check if dim is ANY or ordinal ANY."""
    if isinstance(dim, str):
        # Check if it's a string representation of ANY or ordinal ANY
        return dim.startswith("...") and (len(dim[3:]) == 0 or dim[3:].isdigit())
    return hasattr(dim, "name") and is_ellipses(dim.name)


def is_concrete(dim):
    return not is_any(dim) and not is_ellipses(dim)


def ELLIPSES_MATCH(x, y):
    return is_ellipses(x) and is_ellipses(y) and ND.infer(x).i == ND.infer(y).i


def coalesce(lst, cond=ELLIPSES_MATCH) -> list:
    result = []
    for x in lst:
        if result and cond(result[-1], x):
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
    output = coalesce(output)
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
        """Apply dim-level mapping

        Assigns concrete dims to concrete dims

        Fails if any dim cannot be uniquely assigned.
        """
        src_idxs = [i for i, d in enumerate(self.source) if d == dim]
        tgt_idxs = [self.mapping[src_idx] for src_idx in src_idxs]

        output = None
        for tgt_idx in tgt_idxs:
            if isinstance(tgt_idx, tuple):
                candidate = tuple(self.target[t] for t in tgt_idx)
            else:
                candidate = self.target[tgt_idx]

            if output is None:
                output = candidate
            else:
                if candidate != output:
                    if is_ellipses(output):
                        output = candidate  # override
                    elif not is_ellipses(
                        candidate
                    ):  # neither output nor dandidate are ellipses
                        raise ValueError(
                            f"Cannot uniquely assign dim {dim} to output dim(s) based on {self}, got both {output} and {candidate} as options."
                        )
        return output

    def assign(self, source: Shape):
        output = []
        for src in source:
            if is_ellipses(src):
                output.extend(self.apply(src))
            else:
                output.append(self.apply(src))
        return normalize(output)


def skeleton(shape: tuple[ND | str, ...]):
    """Replace a shape's concrete dims with any dims."""
    replace = {}
    i = 0
    for dim in shape:
        if dim != ELLIPSES and dim not in replace:
            replace[dim] = AnyDim(i)
            i += 1
    return tuple(replace[d] if d != ELLIPSES else d for d in shape)


def match(source: tuple, target: tuple, replace=False) -> Assignment | None:
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
    if replace:
        old_source = copy(source)
        source = skeleton(old_source)
    # Rows = source
    # Cols = target
    ptrs: list[list] = [
        [None for _ in range(len(target) + 1)] for _ in range(len(source) + 1)
    ]
    # Base cases
    ptrs[0][0] = (0, 0)  # True (note that bool(tuple()) == False)
    for i in range(1, len(source) + 1):
        ptrs[i][0] = (-1, 0) if is_ellipses(source[0]) else None
    for j in range(1, len(target) + 1):
        ptrs[0][j] = (0, -1) if is_ellipses(target[0]) else None

    # Dynamic programming
    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if ptrs[i - 1][j - 1] and (
                is_ellipses(source[i - 1])
                or is_ellipses(target[j - 1])
                or source[i - 1] == target[j - 1]
                or is_any(source[i - 1])
                or is_any(target[j - 1])
            ):
                val = (-1, -1)
            elif ptrs[i - 1][j] and is_ellipses(target[j - 1]):
                val = (-1, 0)
            elif ptrs[i][j - 1] and is_ellipses(source[i - 1]):
                val = (0, -1)
            else:
                val = None
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
        if is_ellipses(src):
            output.append(tuple(lst))
        else:
            # Only assign to ELLIPSES if its the only option
            if all(is_ellipses(target[elem]) for elem in lst):
                output.append(lst[0])
            else:
                # Assign to the non-ELLIPSES element
                for elem in lst:
                    if target[elem] != ELLIPSES:
                        output.append(elem)
                        break
    mapping = tuple(output)
    if replace:
        return Assignment(old_source, target, mapping)
    return Assignment(source, target, mapping)


def unify(shapes: Sequence[Shape]):
    """Find the highest-valued shape according to some heuristics."""
    output = shapes[0]
    for target in shapes[1:]:
        a = match(output, target, replace=True)
        if a is None:
            raise ValueError(
                f"Could not unify shapes {shapes}: no assignment possible between {output} and {target}"
            )
        new_shape = []
        for dim in output:
            new_dim = a.apply(dim)
            if new_dim is None:
                raise ValueError(
                    f"Could not unify shapes {shapes}: no matching dim in {a} for {dim}"
                )
            if is_ellipses(dim):
                new_shape.extend(new_dim)
            elif is_any(dim) and is_concrete(new_dim):
                new_shape.append(new_dim)
            elif is_concrete(dim) and is_concrete(new_dim):
                new_shape.append(
                    dim if ND.infer(dim).i > ND.infer(new_dim).i else new_dim
                )
            else:
                new_shape.append(dim)
        output = tuple(new_shape)
    return normalize(output)
