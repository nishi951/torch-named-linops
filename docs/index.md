# torch-named-linops

Welcome to the documentation for `torch-named-linops`, a linear operator
abstraction designed for matrix-free, large-scale numerical computing,
optimization, and machine learning.

## Quickstart

### Installation
```sh
$ pip install torch-named-linops
# or with uv
$ uv add torch-named-linops
```

### Create and Apply a Linear Operator
```python
# Import torch and the core linear operator classes
import torch
from torchlinops import Dense, Dim  # Dense: matrix multiplication, Dim: named dimensions

# Define matrix dimensions
M, N = 3, 7  # M: output size, N: input size

# Create a random weight matrix for our linear operation
weight = torch.randn(M, N)

# Create a Dense linear operator with named dimensions
# Dense performs matrix-vector multiplication using the weight matrix
# Dim("MN") names the weight matrix dimensions,
# ishape=Dim("N") sets the expected input shape,
# oshape=Dim("M") sets the expected output shape.
A = Dense(weight, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))

# Create input data and apply the operator
x = torch.randn(N)  # Random input vector of size N
y = A(x)  # Apply the linear operator (equivalent to A @ x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```

**Expected output:**
```sh
Input shape: torch.Size([7]), Output shape: torch.Size([3])
```

## Features
TODO



## Other Interesting Packages
If you like this package, you may find these other packages interesting as well.
Check them out!
- [SigPy](https://github.com/mikgroup/sigpy/tree/main)
- [MIRTorch](https://github.com/guanhuaw/MIRTorch)
- [SCICO](https://github.com/lanl/scico)
- [matmri](https://gitlab.com/cfmm/matlab/matmri/-/tree/master)
- [einops](https://einops.rocks)
- [torch_linops](https://github.com/cvxgrp/torch_linops)
- [PyLops](https://pylops.readthedocs.io/en/stable/)
- [linear_operator](https://github.com/cornellius-gp/linear_operator)

