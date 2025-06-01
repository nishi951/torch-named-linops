# Getting Started

## A simple example
<!-- name: test_simple_example -->
```python
import torch
from torchlinops import Dense, Dim

# A simple matrix-vector multiply
M, N = (3, 7)
w = torch.randn(M, N)
# A: N -> M represents an MxN dense matrix
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))

# Input to A should have size N
x = torch.randn(N)
y = A(x) 
y2 = A @ x # Alternative syntax

# Adjoint
b = torch.randn(M)
print(A.H)
c = A.H(b)

# Normal
u = torch.randn(N)
v = A.N(u)
```

## Composing linops

<!-- name: test_composition -->
```python
import torch
import torchlinops
from torchlinops import Dense, FFT, Dim

M, N = (3, 7)
w = torch.randn(M, N)
D = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
F = FFT(ndim=1, grid_shapes=(Dim("M"), Dim("K")), batch_shape=Dim())

A = F @ D

x = torch.randn(N)
y = A(x)

# Adjust behavior with config.py
torchlinops.config.reduce_identity_in_normal = False
print(A.N) # Contains Identity in the middle since FFT.H @ FFT = Id
torchlinops.config.reduce_identity_in_normal = True
A.reset_adjoint_and_normal()
print(A.N) # No longer contains Identity



```
## Splitting linops

## Creating a custom linop

