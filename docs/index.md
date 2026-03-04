# Torch Named Linops

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/nishi951/torch-named-linops/test-python.yml)](https://github.com/nishi951/torch-named-linops/actions/workflows/test-python.yml)
[![Codecov](https://img.shields.io/codecov/c/github/nishi951/torch-named-linops)](https://app.codecov.io/gh/nishi951/torch-named-linops)
[![PyPI - Version](https://img.shields.io/pypi/v/torch-named-linops)](https://pypi.org/project/torch-named-linops/)
[![GitHub License](https://img.shields.io/github/license/nishi951/torch-named-linops)](https://www.apache.org/licenses/LICENSE-2.0)

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
# weightshape=Dim("MN") names the weight matrix dimensions (M rows, N columns),
# ishape=Dim("N") sets the expected input shape,
# oshape=Dim("M") sets the expected output shape.
A = Dense(weight, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))

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

- **Named dimensions** -- A dedicated abstraction (`Dim`, `NamedDimension`, `NamedShape`) for naming linear operator dimensions, eliminating shape ambiguity.
- **Automatic adjoints and normals** -- `.H` and `.N` properties to create adjoint ($A^H$) and normal ($A^H A$) operators with correct dimension handling.
- **Operator composition** -- Compose operators with `@` (chaining) and `+` (addition). `Chain`, `Add`, `Concat`, and `Stack` handle dimension matching automatically.
- **Core operator library** -- `Dense`, `Diagonal`, `FFT`, `NUFFT`, `Interpolate`, `ArrayToBlocks`, `Sampling`, and more, all with named dimensions.
- **Multi-GPU splitting** -- Split a single operator across multiple GPUs with `split_linop` and `create_batched_linop`.
- **Complex number support** -- Full support for complex tensors. Adjoint takes the conjugate transpose.
- **Autograd integration** -- Full support for `autograd`-based automatic differentiation through all operators.
- **Iterative solvers** -- Built-in `conjugate_gradients`, `power_method`, and `polynomial_preconditioner` that work directly with named linops.



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

