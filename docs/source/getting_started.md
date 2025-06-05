---
file_format: mystnb
kernelspec:
  name: python3
---
# Getting Started

## A simple example
Start by importing some stuff:
```{code-cell} python
import torch
from torchlinops import Dense, FFT, Dim
```

Create a simple dense matrix-vector multiply.
```{code-cell} python
:tags: [hide-output]
M, N = (3, 7)
w = torch.randn(M, N)
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
print(A)
```

Create a test input and verify that everything checks out:
```{code-cell} python
:tags: [hide-output]
# Input to A should have size N
x = torch.randn(N)
y_ref = w @ x
y = A(x) 
y2 = A @ x # Alternative syntax
print("y_ref matches y:", torch.allclose(y_ref, y))
print("y_ref_matches_y2: ", torch.allclose(y_ref, y2))
```

Compute the adjoint and apply it:
```{code-cell} python
:tags: [hide-output]
b = torch.randn(M)
c = A.H(b)
print(A.H)
```

Compute the normal operator and apply it:
```{code-cell} python
:tags: [hide-output]
u = torch.randn(N)
v = A.N(u)
print(A.N)
```

## Composing linops
Linops can be composed with the `@` operator, creating a `Chain`.

```{admonition} A note on printing linops
When several linops are chained together, they are printed from top to bottom in the order of execution, as this matches the behavior of `nn.Sequential`.
```


```{code-cell} python
:tags: [hide-output]
D = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
F = FFT(ndim=1, grid_shapes=(Dim("M"), Dim("K")), batch_shape=Dim())
A = F @ D
x = torch.randn(N)
y = A(x)
print(A)
```

## Advanced: `torchlinops.config`
### Reducing `Identity` inside normal
```{code-cell} python
:tags: [hide-output]
from torchlinops import config
config.reduce_identity_in_normal = False
print(A.N) # Contains Identity in the middle since FFT.H @ FFT = Id
```

```{code-cell} python
:tags: [hide-output]
config.reduce_identity_in_normal = True
A.reset_adjoint_and_normal() # adjoint and normal are cached by default
print(A.N) # No longer contains Identity
```

## Splitting linops

## Creating a custom linop

