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
  - `ArrayToBlocks`[^*] (similar to PyTorch's [unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html) but in 1D/2D/3D/arbitrary dimensions)
    - Useful for local patch extraction
  - `Interpolate`[^*] (similar to SigPy's
    [interpolate/gridding](https://sigpy.readthedocs.io/en/latest/generated/sigpy.linop.Interpolate.html))
     - Comes with `kaiser_bessel` and (1D) `spline` kernels.
- `.H` and `.N` properties for adjoint $A^H$ and normal $A^HA$ linop creation.
- `Chain` and `Add` for composing linops together.
- `Batch` and `DistributedBatch` wrappers for splitting linops temporally on a
  single GPU, or across multiple GPUs (via `torch.multiprocessing`).
- Full support for complex numbers. Adjoint takes the conjugate transpose.
- Full support for `autograd`-based automatic differentiation.

[^*] Includes a `functional` interface and
[triton](https://github.com/triton-lang/triton) backend for 1D/2D/3D.

### Custom Linops
Custom linops should satisfy the following:
- `@staticmethod` `.fn()`, `.adj_fn()`, and `.normal_fn()`
  - Allows for swapping of functions rather than `bound_methods` in `adjoint()`.
- Calling the constructor of itself via `type(self)` inside `.split_forward()`.
  - This is because of how pytorch handles `copy`-ing parameters. We call the
    constructor rather than deepcopy to avoid making an extra copy of tensors.
    However, `copy` does not propery isolate the new parameter - modifications
    to the parameter will propagate up to the calling function.
- Use of `super().__init__(NS(ishape, oshape))` in the constructor.
  - This initializes the linop's shape
  - May change later to not require `NS`


