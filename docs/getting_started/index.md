# Getting Started

## Installation

Install from PyPI with [uv](https://docs.astral.sh/uv/) or pip:

```console
$ uv add torch-named-linops
# or
$ pip install torch-named-linops
```

To install from source (e.g. for development), clone the repository and install in editable mode:

```console
$ git clone https://github.com/nishi951/torch-named-linops
$ uv add --editable path/to/torch-named-linops
# or
$ pip install -e path/to/torch-named-linops
```

See [Installation](installation.md) for development extras (`[dev]`, `[sigpy]`, `[all]`) and notes on cupy/nccl compatibility.

## A simple example

Operators can infer their shapes automatically. Create a dense matrix-vector multiply with no shapes specified at all:

```python
import torch
from torchlinops import Dense

w = torch.randn(3, 7)  # A (M=3, N=7) matrix
A = Dense(w)  # Shapes inferred from the weight tensor
print(A)
```

```
Dense(((1),) -> ((),))
```

Apply it to a vector — `A(x)` and `A @ x` are equivalent:

```python
x = torch.randn(7)  # Input must have size N
y = A(x)
y2 = A @ x
print(torch.allclose(w @ x, y), torch.allclose(y, y2))
```

```
True True
```

Compute the adjoint and the normal ($A^H A$) operators:

```python
b = torch.randn(3)
c = A.H(b)
print(A.H)

u = torch.randn(7)
v = A.N(u)
print(A.N)
```

```
Dense(((),) -> ((1),))
Dense(((1),) -> ((2),))
```

### Anonymous dimensions

You may notice the printed shapes look like `((1),) -> ((),)`. When shapes are inferred, dimensions get *anonymous* names: `()` means "one unnamed dimension" and parenthesized numbers like `(1)`, `(2)` distinguish different anonymous dimensions from each other. Every operator still tracks these dims and keeps them consistent — they just don't have helpful names yet. This is often good enough for a quick start.

## Naming your dimensions

Anonymous dims compose and match freely against anything, but names pay off as soon as you compose operators, read error messages, or split operators along dimensions. The `Dim()` convenience function creates a tuple of dimension names from a compact string:

```python
from torchlinops import Dim

Dim("MN")  # -> ("M", "N")  — each uppercase letter starts a new dimension
Dim("NxNy")  # -> ("Nx", "Ny") — uppercase + lowercase letters form one name
Dim("M")  # -> ("M",)
Dim("")  # -> ()           — empty string for scalar/batch dims
```

You can also pass a plain tuple `("M", "N")` anywhere `Dim("MN")` is accepted. Both forms are equivalent.

For example, the dense operator above with fully named dimensions:

```python
import torch
from torchlinops import Dense, Dim

M, N = 3, 7
w = torch.randn(M, N)
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
print(A)
```

```
Dense((N,) -> (M,))
```

!!! tip "weightshape vs ishape/oshape"
    When creating operators like `Dense`, you'll encounter both:

    - **`weightshape`** — describes the dimensions of the *matrix* itself (the storage). For a matrix with shape `(M, N)`, this is `Dim("MN")`.
    - **`ishape` / `oshape`** — describe what the operator maps *from* and *to*. These are the dimensions of the input and output vectors.

    For example, with a weight matrix `W` of shape `(3, 7)`:
    ```python
    A = Dense(W, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
    #                   ↑ matrix dims (M=3, N=7)       ↑ input (7)  ↑ output (3)
    ```

## Composing linops

Linops can be composed with the `@` operator, creating a `Chain`.

!!! note "A note on printing linops"
    When several linops are chained together, they are printed from top to bottom in the order of execution, as this matches the behavior of `nn.Sequential`.

```python
import torch
from torchlinops import Dense, FFT, Dim

M, N, K = 3, 7, 5
w = torch.randn(M, N)
D = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
F = FFT(ndim=1, grid_shapes=(Dim("M"), Dim("K")))
A = F @ D
x = torch.randn(N)
y = A(x)  # FFT(w @ x)
print(A)
```

```
Chain(
    Dense((N,) -> (M,))
    FFT((..., M) -> (..., K))
)
```

Note that the output dim of `Dense` (`M`) matches the input grid dim of `FFT`. Names must line up for operators to compose: the chain checks at construction time that each operator's output shape equals the next operator's input shape. `(...)` is a wildcard batch dimension — here it matched zero batch dims.

Most operators omitting shape arguments fall back to sensible defaults: `FFT(ndim=1)` names its grids `Nx` and `Kx` automatically (`(..., Nx) -> (..., Kx)`), and batch dimensions default to `...`. Specify `grid_shapes=(...)` and `batch_shape=...` explicitly when you need to control the naming, e.g. to match other operators' dimensions.

## Where to go next

- **[Splitting and batching linops](splitting.md)** — chunk large operators along named dimensions, e.g. to process data too big for one GPU.
- **[Creating custom linops](../howto/custom_linops.md)** — subclass `NamedLinop` to add your own operator (see also the [custom linops tutorial](../tutorials/custom_linop.md)).
- **[Configuration options](../explanations/config.md)** — global toggles like `reduce_identity_in_normal`, `cache_adjoint_normal`, and opt-in shape inference in `Chain`/`Add`.
- **[How named dimensions propagate](../explanations/named_dims_propagation.md)** — how shapes are inferred, renamed, and incremented as you build, compose, adjoint, and normalize operators.
