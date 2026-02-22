# Design Notes

Various notes on the design of this package. For the full description of each abstraction, see [Named Abstractions](named_abstractions.md).

## Why `fn` and `adj_fn` are static methods

The forward and adjoint functions on `NamedLinop` are declared as `@staticmethod` rather than as regular instance methods. This is a deliberate design choice that enables the adjoint mechanism.

When `adjoint()` is called, it creates a shallow copy of the linop and swaps the two functions:

```python
adj.fn, adj.adj_fn = adj.adj_fn, adj.fn
```

If `fn` and `adj_fn` were regular bound methods, this swap would not work correctly -- the methods would remain bound to their original implementations. Static methods are unbound functions stored as plain attributes on the class (or instance), so they can be freely reassigned.

The tradeoff is that every `fn` and `adj_fn` signature takes the linop as its first argument (`linop` rather than `self`), which is slightly unconventional. This is the cost of enabling zero-overhead adjoint creation.

## Why adjoint and normal caches use lists

Cached adjoints and normals are stored as single-element lists:

```python
self._adjoint = [adj]   # Not self._adjoint = adj
self._normal = [normal]
```

This is necessary because `NamedLinop` inherits from `nn.Module`. PyTorch's `nn.Module.__setattr__` intercepts any attribute assignment and, if the value is an `nn.Module`, registers it as a submodule. If we stored `self._adjoint = adj` directly, PyTorch would:

1. Register the adjoint as a submodule, causing its parameters to appear in `self.parameters()`.
2. Create circular references (`A._adjoint = A.H`, and `A.H._adjoint = A`), leading to infinite recursion during parameter traversal.

Wrapping in a list hides the module from PyTorch's registration logic while still providing O(1) access.

## Pickle-ability and multiprocessing

Several design choices are motivated by the need for linops to be picklable, which is required for `torch.multiprocessing`:

- **`NormalFunctionLookup`**: The `normal()` method needs to create a linop whose forward pass calls `normal_fn`. The natural approach would be a lambda, but lambdas are not picklable. Instead, a helper class `NormalFunctionLookup` stores a reference to the original linop and provides named methods that can be pickled.

- **`new_normal_adjoint`**: Similarly, the adjoint of a normal operator needs a custom `adjoint()` method. This is defined as a top-level function (not a lambda or closure) to maintain pickle-ability, and bound to the normal linop via `functools.partial`.

## Shape matching with wildcards

The `_matching.py` module implements shape compatibility checking with wildcard support. It uses a dynamic programming algorithm (similar to regex matching) to handle:

- **`...` (ELLIPSES)**: Can match zero or more dimensions in either shape. For example, `("A", "...", "C")` is compatible with `("A", "B1", "B2", "C")`.
- **`()` (ANY)**: Matches exactly one dimension of any name. For example, `("A", "()")` is compatible with `("A", "B")`.

This is used in two contexts:

1. **`isequal()`**: Checks if two shapes are value-compatible (same dims in compatible positions). Used for verifying that composed linops have matching intermediate shapes.
2. **`iscompatible()`**: Checks if two shapes are length-compatible (ignoring specific dim names). Used when updating shapes in a `NamedDimCollection`, where the new shape must have the same structure as the old one.

## Shallow copy as the primary reuse mechanism

The library favors `copy.copy()` (shallow copy) over `copy.deepcopy()` as the primary way to create derived operators (adjoints, normals, splits). This is because:

- Linop data (weights, buffers) can be very large (e.g., multi-GB interpolation tables). Duplicating this data for every adjoint or split would be prohibitively expensive.
- A shallow copy shares the data but gets its own shape, function references, and cache state. This is exactly what's needed for an adjoint: same data, different interpretation.
- When true data independence is needed (e.g., for multi-GPU placement), the library provides `memory_aware_deepcopy` as an explicit opt-in. See [Copying Linops](copying_linops.md).
