# Getting Started

## Understanding `Dim()`

Before diving in, a quick note on `Dim()`. It is a convenience function for creating named dimension tuples from a compact string:

```python
from torchlinops import Dim

Dim("MN")     # -> ("M", "N")  — each uppercase letter starts a new dimension
Dim("NxNy")   # -> ("Nx", "Ny") — uppercase + lowercase letters form one name
Dim("M")      # -> ("M",)
Dim("")        # -> ()          — empty string for scalar/batch dims
```

You can also pass a plain tuple `("M", "N")` anywhere `Dim("MN")` is accepted. Both forms are equivalent.

!!! tip "weightshape vs ishape/oshape"
    When creating operators like `Dense`, you'll encounter both:

    - **`weightshape`** — describes the dimensions of the *matrix* itself (the storage). For a matrix with shape `(M, N)`, this is `Dim("MN")`.
    - **`ishape` / `oshape`** — describe what the operator maps *from* and *to*. These are the dimensions of the input and output vectors.

    For example, with a weight matrix `W` of shape `(3, 7)`:
    ```python
    A = Dense(W, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
    #                   ↑ matrix dims (M=3, N=7)       ↑ input (7)  ↑ output (3)
    ```

## A simple example
Start by importing some stuff:
```python
import torch
from torchlinops import Dense, FFT, Dim
```

Create a simple dense matrix-vector multiply.
```python
M, N = (3, 7)
w = torch.randn(M, N)
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
print(A)
```

Create a test input and verify that everything checks out:
```python
# Input to A should have size N
x = torch.randn(N)
y_ref = w @ x
y = A(x) 
y2 = A @ x # Alternative syntax
print("y_ref matches y:", torch.allclose(y_ref, y))
print("y_ref_matches_y2: ", torch.allclose(y_ref, y2))
```

Compute the adjoint and apply it:
```python
b = torch.randn(M)
c = A.H(b)
print(A.H)
```

Compute the normal operator and apply it:
```python
u = torch.randn(N)
v = A.N(u)
print(A.N)
```

## Composing linops
Linops can be composed with the `@` operator, creating a `Chain`.

!!! note "A note on printing linops"
    When several linops are chained together, they are printed from top to bottom in the order of execution, as this matches the behavior of `nn.Sequential`.


```python
D = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
F = FFT(ndim=1, grid_shapes=(Dim("M"), Dim("K")), batch_shape=Dim())
A = F @ D
x = torch.randn(N)
y = A(x)
print(A)
```

## Advanced: `torchlinops.config`
### Reducing `Identity` inside normal
```python
from torchlinops import config
config.reduce_identity_in_normal = False
print(A.N) # Contains Identity in the middle since FFT.H @ FFT = Id
```

```python
config.reduce_identity_in_normal = True
print(A.N) # No longer contains Identity
```

!!! info "What does `reduce_identity_in_normal` do?"
    When computing the normal operator $A^H A$, some operators have the property that $F^H F = I$ (e.g., FFT with orthonormal normalization). By default, the library simplifies these to `Identity` operators in the normal chain.

    - **`True`** (default): Simplifies `FFT.H @ FFT` to `Identity` in the normal operator, producing a more compact operator graph.
    - **`False`**: Keeps the full chain, which can be useful for debugging or when you need to inspect the exact computation.

    Note that you must call `reset_adjoint_and_normal()` after changing this setting to clear the cached operators.

### Temporarily modifying config

Use `config.using()` as a context manager to temporarily modify config values:

```python
from torchlinops import config

with config.using(reduce_identity_in_normal=False):
    # reduce_identity_in_normal is False here
    print(A.N)
# original value restored
```

You can modify multiple values at once:

```python
with config.using(reduce_identity_in_normal=False, log_device_transfers=False):
    # multiple values temporarily changed
    pass
```

### Shape inference

When composing linops, you can enable automatic shape inference to reduce boilerplate. This is especially useful for chains where intermediate shapes can be inferred from the composition structure.

```python
from torchlinops import FFT, Sampling, Diagonal, config
from torchlinops.linops.chain import Chain

# Create operators with unspecified or partially-specified shapes
F = FFT(ndim=2, batch_shape=("C",))  # oshape: (C, Kx, Ky)
S = Sampling(idx, input_size=(64, 64), batch_shape=("C",), output_shape=("K",))  # ishape: (C, ...)
D = Diagonal(weights)                   # ioshape: (...)

# Without shape inference, you must specify all shapes manually
# With shape inference, shapes propagate automatically
with config.using(shape_inference=True):
    model = Chain(F, S, D)
    print(model)
```

Output:
```
Chain(
  FFT(('C', 'Nx', 'Ny') -> ('C', 'Kx', 'Ky'))
  Sampling(('C', 'Kx', 'Ky') -> ('C', 'K'))
  Diagonal(('C', 'K') -> ('C', 'K'))
)
```

Notice how `Sampling`'s input shape was resolved from `(C, ...)` to `(C, Kx, Ky)`, and `Diagonal`'s shape was resolved from `(...)` to `(C, K)`.

Shape inference also works with `Add` for regularization patterns:

```python
from torchlinops import Identity

# Common pattern: A^H A + lambda * I
with config.using(shape_inference=True):
    normal = model.N + 0.01 * Identity()  # Identity's shape inferred from model.N
    print(normal)
```

Output:
```
Add(
  Chain(...)  # model.N with shape (('C', 'Nx', 'Ny'),) -> (('C', 'Nx1', 'Ny1'),)
  Scalar(Identity(('C', 'Nx', 'Ny') -> ('C', 'Nx', 'Ny')))
)
```

!!! note "Shape inference behavior"
    - Wildcards (`...` and `()`) in shapes are resolved using compatibility checking
    - Partially-specified shapes like `("C", "...")` are resolved to match concrete shapes
    - Shape inference is **disabled by default** and must be explicitly enabled
    - Existing validation still applies — incompatible shapes will raise errors

## Splitting linops

Linops can be split across sub-problems (e.g., for multi-GPU processing) using `split_linop`. See [Multi-GPU Execution](../explanations/multi_gpu.md) for the full explanation.

```python
from torchlinops import split_linop
# splits = split_linop(A, ...)  # Splits A into sub-operators
```

## Creating a custom linop

You can create your own `NamedLinop` by subclassing it and implementing `fn()` and `adj_fn()` as static methods. See the [Custom Linops How-To Guide](../howto/custom_linops.md) for the full requirements and a complete example.
