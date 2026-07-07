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

## Composite linop execution model

The `Add`, `Concat`, `Stack`, and `Chain` classes are **composite linops** — they hold other linops as children and coordinate their execution. Each manages its own `linops` property and controls parallelism through `threaded` and `num_workers` constructor parameters.

### Why linops is a property

Each composite linop manages sub-linops through a `linops` property rather than a direct attribute. This design choice is intentional:

```python
class Add(NamedLinop):
    @property
    def linops(self):
        return self._linops

    @linops.setter
    def linops(self, new_linops):
        self._linops = new_linops
```

The property ensures that when `linops` is reassigned, any dependent state is kept in sync. Unlike the old `Threadable` mixin (removed in the stream-sync refactor), the current model **does not** create shallow copies of shared linops or set up input listeners. Shared linops (e.g., `Add(A, A)`) are used directly — each thread receives the same linop object but an independent execution context.

### Why `__setattr__` bypass is needed

PyTorch's `nn.Module.__setattr__` intercepts all attribute assignments. When you set `self.linops = nn.ModuleList([...])`, PyTorch would register each linop as a submodule. The `__setattr__` override on each composite class ensures that `linops` assignment goes through the property descriptor instead:

```python
def __setattr__(self, name, value):
    if name == "linops":
        type(self).linops.fset(self, value)  # Use descriptor
    else:
        super().__setattr__(name, value)
```

### Parallel execution via `parallel_execute`

When `threaded=True` (the default), composite linops use `parallel_execute()` from `torchlinops.linops.schedule` to run child linops in a `ThreadPoolExecutor`. This is beneficial when sub-linops release the GIL (e.g., PyTorch tensor operations). The `num_workers` parameter controls the maximum number of threads; if `None`, it defaults to the number of children.

When `threaded=False`, children run sequentially in a simple loop. This is useful for debugging or when thread overhead outweighs benefits.

```python
# Parallel (default)
add = Add(A, B, C)  # threaded=True by default

# Sequential
add = Add(A, B, C, threaded=False)

# Limited parallelism
add = Add(A, B, C, threaded=True, num_workers=2)
```

Shared linops are used directly without copying:

```python
A = Dense(weight, ...)
add = Add(A, A)  # Same A object in both slots

# linops[0] and linops[1] ARE the same object
assert add.linops[0] is add.linops[1]

# They share the same weight data (obviously, since they're the same object)
assert add.linops[0].weight.data_ptr() == add.linops[1].weight.data_ptr()
```

See [Multi-GPU Execution](multi_gpu.md) for the synchronization mechanism (`SyncContext`, CUDA events, and streams) that makes parallel containers work safely.

## Gotchas and Pitfalls

### Shallow copy shares weight data

When you access `A.H` (adjoint) or `A.N` (normal), the returned operator is a shallow copy that shares the same weight tensors as the original:

```python
A = Dense(weight, ...)
adj = A.H
adj.weight is A.weight  # True! They share the same data

# Modifying weights in one affects the other
adj.weight.data.fill_(0)
print(A.weight.sum())  # 0.0
```

This is intentional—modifying the operator should update both forward and adjoint consistently. If you need an independent copy, use `torchlinops.copying.memory_aware_deepcopy(A)`.

### View relationships can break with deepcopy

If your linop uses views into a larger tensor (e.g., slices of a shared buffer), a naive `copy.deepcopy()` will allocate separate storage for each view, potentially doubling memory usage. Use `memory_aware_deepcopy()` to preserve view relationships:

```python
from torchlinops.copying import memory_aware_deepcopy

# Preserves view relationships
A_copy = memory_aware_deepcopy(A)
```

### Complex numbers: adjoint includes conjugate

For complex-valued operators, the adjoint includes the complex conjugate:

```python
# For complex weight matrix W:
# Forward: y = W @ x
# Adjoint: y = W.conj().T @ x

# This is automatically handled by Dense and other operators,
# but if you're implementing a custom linop, remember to include .conj()
```

### Linops are non-trainable by default

Linops use `nn.Parameter` with `requires_grad=False` by default, since linear operators in optimization problems are typically fixed. To make weights trainable:

```python
weight = nn.Parameter(torch.randn(M, N), requires_grad=True)
A = Dense(weight, ...)
# Now A supports autograd
```

### Pickle requirements for multiprocessing

If you plan to use linops with `torch.multiprocessing`, they must be picklable. The library uses specific patterns (static methods, `NormalFunctionLookup` class) to ensure this. Avoid using lambdas or closures in custom linop implementations.
