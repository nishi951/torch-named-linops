# Triton kernels reference a module-level `cast` alias that Inductor drops

## Observed

CUDA `torch.compile` of `ArrayToBlocks`, `BlocksToArray`, `functional.unfold`, and `functional.fold` fails while Inductor compiles the triton kernel source:

```
torch._inductor.exc.InductorError: CompilationError: at 39:16:
    # Convert types
    x_nblocks = cast(x_nblocks, tl.uint64)
                ^
NameError('cast is not defined')
```

Eager execution is fine, and the same functions compile on CPU (2.2–5.3× speedups), so this is specific to triton kernel recompilation.

## Minimal repro

```python
import torch
from torchlinops.linops import ArrayToBlocks

op = ArrayToBlocks((8, 8), (4, 4), (4, 4))
x = torch.randn(2, 8, 8, dtype=torch.complex64, device="cuda")
torch.compile(lambda t: op(t))(x)
# InductorError ... NameError('cast is not defined')
```

## Where

- `src/torchlinops/functional/_unfold/unfold.py` (kernel bodies `_unfold1d/2d/3d`; `x_nblocks = cast(x_nblocks, tl.uint64)`)
- `src/torchlinops/functional/_unfold/fold.py` (analogous `cast` calls)
- alias definition: `from .casting import scalar_cast as cast` in those modules; Inductor regenerates kernel source without the module-level alias in scope.

## Suggested fix

Stop depending on the Python-level alias inside kernel bodies. Either:

- use `tl.cast(x, tl.uint64)` / target-dtype `.to(...)` directly inside kernels, or
- pass the callable explicitly as a triton constexpr/global to the kernel invocation.

Apply to both unfold and fold; audit `functional/_interp/` for the same pattern.

## Value

Unblocks CUDA compile for ArrayToBlocks/BlocksToArray and the four functional kernels (currently the only eager crashes in the matrix are unrelated autograd gaps). Also makes their eager-vs-compiled parity testable on GPU.

## Verification idea

Add a CUDA-marked test that `torch.compile(op)(x)` equals eager for a small unfold/fold case. Do not require `fullgraph` until the wrapper logging issue is fixed.
