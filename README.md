# torch-named-linops

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/nishi951/torch-named-linops/test-python.yml)](https://github.com/nishi951/torch-named-linops/actions/workflows/test-python.yml)
[![Codecov](https://img.shields.io/codecov/c/github/nishi951/torch-named-linops)](https://app.codecov.io/gh/nishi951/torch-named-linops)
[![PyPI - Version](https://img.shields.io/pypi/v/torch-named-linops)](https://pypi.org/project/torch-named-linops/)
[![GitHub License](https://img.shields.io/github/license/nishi951/torch-named-linops)](https://www.apache.org/licenses/LICENSE-2.0)

A flexible linear operator abstraction implemented in PyTorch.

``` sh
$ pip install torch-named-linops
```

## Selected Feature List
- A dedicated abstraction for naming linear operator dimensions.
- A set of core linops, including:
  - `Dense`
  - `Diagonal`
  - `FFT`
  - `ArrayToBlocks`[^1] (similar to PyTorch's [unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html) but in 1D/2D/3D/arbitrary dimensions)
    - Useful for local patch extraction
  - `Interpolate`[^1] (similar to SigPy's
    [interpolate/gridding](https://sigpy.readthedocs.io/en/latest/generated/sigpy.linop.Interpolate.html))
     - Comes with `kaiser_bessel` and first-order `spline` kernels.
- `.H` and `.N` properties for adjoint $A^H$ and normal $A^HA$ linop creation.
- `Chain` and `Add` for composing linops together.
- Splitting a single linop across multiple GPUs.
- Full support for complex numbers. Adjoint takes the conjugate transpose.
- Full support for `autograd`-based automatic differentiation.

[^1]: Includes a `functional` interface and [triton](https://github.com/triton-lang/triton) backend for 1D/2D/3D.


## Other Packages
This package was heavily inspired by a few other influential packages. In no particular order:
- [einops](https://einops.rocks): named dimensions/naming things in general.
- [sigpy](https://github.com/mikgroup/sigpy): the linop abstraction and the idea of having dedicated adjoint and normal properties. Also inspired the NUFFT, Interpolate, and ArrayToBlocks/BlocksToArray operators.
- [torch_linops](https://github.com/cvxgrp/torch_linops): another linop abstraction. Geared more towards optimization.

