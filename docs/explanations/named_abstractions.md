# Named Abstractions

This page describes the core abstraction stack that underpins `torch-named-linops`, from the atomic `NamedDimension` up through `NamedLinop` itself.

## NamedDimension

`NamedDimension` is the atomic unit of shape specification. It is a frozen dataclass with two fields:

- **`name`** (str): The human-readable label, e.g. `"Nx"`, `"Batch"`, `"K"`.
- **`i`** (int, default 0): A numeric index that distinguishes multiple dimensions with the same base name.

```python
from torchlinops.nameddim import NamedDimension as ND

h = ND("H")       # H (i=0, not printed)
h1 = ND("H", 1)   # H1
```

### String interoperability

`NamedDimension` is designed to be interchangeable with plain strings in most contexts. Its `__eq__` and `__hash__` are based on `repr()`, so:

```python
ND("A") == "A"          # True
ND("A", 1) == "A1"      # True
{ND("A"): 10}["A"]      # Works -- same hash
```

This means you can use plain strings like `"Nx"` when constructing shapes, and they will be automatically converted to `NamedDimension` objects via the `ND.infer()` classmethod.

### The `Dim()` convenience parser

For compact shape specification, the `Dim()` function parses a single string into a tuple of dimension names based on simple rules (each new uppercase letter starts a new dim):

```python
from torchlinops import Dim

Dim("ABCD")       # ('A', 'B', 'C', 'D')
Dim("NxNyNz")     # ('Nx', 'Ny', 'Nz')
Dim("A1B2Kx1Ky2") # ('A1', 'B2', 'Kx1', 'Ky2')
```

### Special dimensions

Two special dimension names have wildcard semantics:

- **`"..."`** (ELLIPSES): Matches any number of dimensions (including zero). Analogous to Python's `Ellipsis` in slicing.
- **`"()"`** (ANY): Matches exactly one dimension of any name. A single-position wildcard.

These are used internally for flexible shape matching during composition and updates.

### Generating collision-free names

When forming the normal operator $A^H A$, the output dimensions must be distinct from the input dimensions to avoid ambiguity. The `next_unused(tup)` method handles this:

```python
a = ND("A")
a.next_unused(("A", "B"))   # A1 (since "A" is taken)
a.next_unused(("A", "A1"))  # A2
```

---

## NamedDimCollection

`NamedDimCollection` is a container that manages multiple **named shapes** over a shared pool of dimensions. It is the mechanism that ensures cross-shape consistency when dimensions are renamed or updated.

### Shared-storage design

Internally, a `NamedDimCollection` stores:

- **`_dims`**: A flat list of `NamedDimension` objects (the shared pool).
- **`idx`**: A dictionary mapping shape names (e.g. `"ishape"`, `"oshape"`) to tuples of integer indices into `_dims`.

For example, given shapes `ishape = (A, B)` and `oshape = (B, C)`:

```
_dims = [A, B, C]
idx   = {"ishape": (0, 1), "oshape": (1, 2)}
```

Both shapes share dimension `B` at index 1. If `B` is renamed to `E`, it changes in `_dims[1]` and is immediately reflected in both `ishape` and `oshape`.

### Why this matters

Without shared storage, renaming a dimension in one shape could silently leave the other shape out of sync. The index-based design makes cross-shape consistency automatic: there is a single source of truth for each dimension.

### Attribute-style access

Shapes can be read and written as attributes:

```python
from torchlinops.nameddim import NamedDimCollection

c = NamedDimCollection(shape1=("A", "B"), shape2=("B", "C"))
c.shape1          # (A, B)
c.shape1 = ("D", "E")  # Renames A->D, B->E across ALL shapes
c.shape2          # (E, C)  -- B was renamed to E here too
```

Shape updates use the `iscompatible()` matcher to verify that the new shape is length-compatible with the old one (accounting for `...` and `()` wildcards).

---

## NamedShape

`NamedShape` inherits from `NamedDimCollection` and specializes it for linear operators. It always contains two distinguished shapes:

- **`ishape`**: The input dimensions of the operator.
- **`oshape`**: The output dimensions of the operator.

Additional shapes can be stored for operator-specific metadata (e.g., `batch_shape`, `grid_shape`).

### Construction

```python
from torchlinops.nameddim import NamedShape as NS

# Full specification
s = NS(("Nx", "Ny"), ("Kx", "Ky"))   # (Nx, Ny) -> (Kx, Ky)

# Diagonal shortcut: oshape = ishape
s = NS(("A", "B"))                     # (A, B) -> (A, B)

# Pass-through: accepts another NamedShape
s2 = NS(s)                             # copies s
```

### Adjoint shape: `.H`

The adjoint swaps input and output:

```python
s = NS(("Nx", "Ny"), ("Kx", "Ky"))
s.H   # (Kx, Ky) -> (Nx, Ny)
```

### Normal shape: `.N`

The normal operator $A^H A$ maps from `ishape` back to `ishape`. But since the intermediate `oshape` dimensions must be distinct from the input dimensions, `.N` generates new output dim names using `next_unused()`:

```python
s = NS(("A", "B"), ("C", "D"))
s.N   # (A, B) -> (A1, B1)
```

The output dims `A1`, `B1` are the first unused variants of the input dims `A`, `B`.

### Shape arithmetic

Shapes can be added to concatenate their components, which is useful when combining batch dimensions with spatial dimensions:

```python
batch = NS(("Batch",), ("Batch",))
spatial = NS(("Nx", "Ny"), ("Kx", "Ky"))
full = batch + spatial   # (Batch, Nx, Ny) -> (Batch, Kx, Ky)
```

---

## NamedLinop

`NamedLinop` is the base class for all linear operators in the library. It inherits from `torch.nn.Module`, so linops are first-class PyTorch modules with full support for parameters, buffers, GPU placement, and serialization.

### Shape management

Each linop holds a `NamedShape` (via the private `_shape` attribute) and exposes it through properties:

```python
A.ishape   # Input dimensions
A.oshape   # Output dimensions
A.shape    # The full NamedShape
A.dims     # Set of all dimensions (ishape union oshape)
```

Setting `ishape` or `oshape` mutates the underlying `NamedShape`, which propagates changes to any shared dimensions.

### The function interface

The forward, adjoint, and normal operations are defined as **static methods**:

```python
@staticmethod
def fn(linop, x: Tensor, /) -> Tensor:
    """Forward: y = A(x)"""
    ...

@staticmethod
def adj_fn(linop, x: Tensor, /) -> Tensor:
    """Adjoint: x = A^H(y)"""
    ...

@staticmethod
def normal_fn(linop, x: Tensor, /) -> Tensor:
    """Normal: z = A^H(A(x))"""
    return linop.adj_fn(linop, linop.fn(linop, x))
```

The `forward()` method wraps `fn()` with optional CUDA stream execution:

```python
def forward(self, x):
    if self.stream is not None:
        with torch.cuda.stream(self.stream):
            y = self.fn(self, x)
        x.record_stream(self.stream)
        return y
    return self.fn(self, x)
```

Why static methods? Because `adjoint()` creates the adjoint by simply **swapping** `fn` and `adj_fn` on a shallow copy. If these were bound methods, swapping would not work -- the methods would still be bound to the original instance. Static methods are unbound functions that can be freely reassigned. See [Design Notes](design_notes.md) for more on this decision.

### Lazy cached operators

Accessing `.H` or `.N` creates the derived operator on first access and caches it:

```python
A.H   # Creates adjoint on first call, returns cached version thereafter
A.N   # Creates normal on first call, returns cached version thereafter
```

The cache is stored as a single-element list (e.g. `self._adjoint = [adj]`) rather than as a direct attribute. This prevents `nn.Module.__setattr__` from registering the derived operator as a submodule, which would cause parameter duplication and circular references.

The adjoint and normal caches link back to each other: `A.H._adjoint = [A]`, so `A.H.H` returns the original `A` without creating a new copy.

### Adjoint creation

The default `adjoint()` method works by:

1. Shallow-copying the linop (shares all parameters/buffers).
2. Flipping the shape: `adj._shape = adj._shape.H`.
3. Swapping the functions: `adj.fn, adj.adj_fn = adj.adj_fn, adj.fn`.
4. Swapping the split functions: `adj.split, adj.adj_split = adj.adj_split, adj.split`.

This means that for most linops, you only need to implement `fn` and `adj_fn` -- the adjoint operator is automatically constructed from those two functions.

### Normal operator creation

The default `normal()` method creates a linop whose forward pass calls `normal_fn()`. It uses a `NormalFunctionLookup` helper class (rather than lambdas) to maintain pickle-ability, which is required for multiprocessing.

Many linops override `normal()` with optimized implementations:

- **FFT**: $F^H F = I$ (identity), since the DFT with orthonormal normalization is unitary.
- **Diagonal**: $(D^H D)(x) = |w|^2 \odot x$, a single elementwise multiply.
- **Chain**: $(\ldots B A)^H (\ldots B A)$ is computed by folding `normal()` calls through the chain, enabling [Toeplitz embedding](https://en.wikipedia.org/wiki/Toeplitz_matrix) and other optimizations.

### Operator algebra

`NamedLinop` overloads Python operators for natural composition:

| Syntax | Result | Semantics |
|--------|--------|-----------|
| `A @ B` | `Chain(B, A)` | Composition: first apply $B$, then $A$ |
| `A + B` | `Add(A, B)` | Summation: $y = A(x) + B(x)$ |
| `c * A` | `Scalar(c) @ A` | Scalar multiplication |
| `A @ x` | `A(x)` | Application to a tensor |

Note that `Chain` stores linops in **execution order** (inner-to-outer), so `A @ B` creates `Chain(B, A)` -- `B` is applied first.

### Splitting

Every linop can implement `split_forward(ibatch, obatch)` to support decomposition into sub-linops along named dimensions. The default `split()` static method translates a tile dictionary (mapping dim names to slices) into the `ibatch`/`obatch` format and delegates to `split_forward`. See [Multi-GPU Splitting](multi_gpu.md) for how this is used to distribute computation.

### Size reporting

The `size(dim)` method lets each linop report the concrete size of dimensions it "owns". This is used by the splitting machinery to determine how many tiles to create. Linops that don't determine a dimension's size return `None`, and the splitting system aggregates sizes across the full operator chain.
