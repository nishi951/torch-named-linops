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
A.reset_adjoint_and_normal() # adjoint and normal are cached by default
print(A.N) # No longer contains Identity
```

## Splitting linops

Linops can be split across sub-problems (e.g., for multi-GPU processing) using `split_linop`. See [Multi-GPU Splitting](../explanations/multi_gpu.md) for the full explanation.

```python
from torchlinops import split_linop
# splits = split_linop(A, ...)  # Splits A into sub-operators
```

## Creating a custom linop

You can create your own `NamedLinop` by subclassing it and implementing `fn()` and `adj_fn()` as static methods. See the [Custom Linops How-To Guide](../howto/custom_linops.md) for the full requirements and a complete example.
