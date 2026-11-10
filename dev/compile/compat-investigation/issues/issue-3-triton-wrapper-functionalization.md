# `triton_kernel_wrapper_functional` survives functionalization for interp kernels

## Observed

CUDA `torch.compile` of `Interpolate`, `NUFFT`, `functional.grid`, and `functional.ungrid` fails in Inductor:

```
torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed
```

This also fails compiled backward for `Interpolate`/`NUFFT`, and makes the realistic `NUFFT @ Diagonal` pipeline uncompilable in every mode (including the compiled-inners pattern). CPU compilation of the same operators works.

## Minimal repro

```python
import torch
from torchlinops import NUFFT
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs

locs = get_valid_locs((8,), (16, 16), 2, 4.0, "cpu", centered=True).cuda()
op = NUFFT(locs.clone(), (16, 16), output_shape=("R",), width=4.0, oversamp=1.25)
x = torch.randn(16, 16, dtype=torch.complex64, device="cuda")
torch.compile(lambda t: op(t))(x)
# InductorError: AssertionError: triton_kernel_wrapper_functional was not removed
```

## Where

`src/torchlinops/functional/_interp/grid.py` / `ungrid.py` (triton kernel launch path), reached from `Interpolate.fn` and the NUFFT family. Likely interaction between the custom kernel wrapper and `torch.compile`'s functionalization pass (kernel wrapper with mutated/aliased output buffer).

## Suggested fix

Investigate the launch pattern against `torch.library.custom_op` guidance:

- wrap the kernel in `torch.library.custom_op` with fake/meta implementations so Inductor treats it as an opaque op instead of trying to decompose the wrapper;
- or return a fresh output tensor rather than passing/aliasing an output buffer into the kernel wrapper.

Confirm with `TORCH_LOGS=aot_graphs` which graph still contains `triton_kernel_wrapper_functional`.

## Value

Unblocks the largest capability gap in the investigation: CUDA-compiled NUFFT/Interpolate forward *and* backward, and therefore realistic MRI-recon pipeline compilation. Higher effort than the logging/cast fixes; ranked after them.
