# torch-named-linops

A flexible linear operator abstraction implemented in PyTorch.

Heavily inspired by [einops](https://einops.rocks).

Unrelated to the (also good) [torch_linops](https://github.com/cvxgrp/torch_linops)

## Selected Feature List
- Dedicated abstraction for naming linear operator dimensions.
- A set of core linops, including:
  - `Dense`
  - `Diagonal`
  - `FFT`
  - `ArrayToBlocks`[^1] (similar to PyTorch's [unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html) but in 1D/2D/3D/arbitrary dimensions)
    - Useful for local patch extraction
  - `Interpolate`[^1] (similar to SigPy's
    [interpolate/gridding](https://sigpy.readthedocs.io/en/latest/generated/sigpy.linop.Interpolate.html))
     - Comes with `kaiser_bessel` and (1D) `spline` kernels.
- `.H` and `.N` properties for adjoint $A^H$ and normal $A^HA$ linop creation.
- `Chain` and `Add` for composing linops together.
- `Batch` and `DistributedBatch` wrappers for splitting linops temporally on a
  single GPU, or across multiple GPUs (via `torch.multiprocessing`).
- Full support for complex numbers. Adjoint takes the conjugate transpose.
- Full support for `autograd`-based automatic differentiation.

[^1]: Includes a `functional` interface and [triton](https://github.com/triton-lang/triton) backend for 1D/2D/3D.

## Installation
### Via `pip`

``` sh
$ pip install torch-named-linops
```

### From source (recommended for developers)
1. Clone the repo with `git clone`
2. Run `pip install -e .` from the root directory.
  - Or `uv add path/to/cloned/repo`

3. Pull upstream changes as required.

### Via `pip`'s git integration (deprecated)
Run the following, replacing `<TAG>` with the appropriate version (e.g. `0.3.7``)

- http version:
```sh
$ pip install git+https://github.com/nishi951/torch-named-linops.git@<TAG>
```

- ssh version:
``` sh
$ pip install git+ssh://git@github.com/nishi951/torch-named-linops.git@<TAG>
```


