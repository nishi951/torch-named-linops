# Functional unfold/fold/grid/ungrid kernels are not differentiable

## Observed

Dead `backward()` through the public functional API fails in **eager** mode, i.e. this is not a `torch.compile` regression:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

Confirmed for `functional.unfold`, `functional.fold`, `functional.grid`, `functional.ungrid`. Consequence for the compile investigation: these kernels cannot participate in training loops at all yet (goal 2 of the investigation), independent of the triton blockers.

## Minimal repro

```python
import torch
from torchlinops.functional import unfold

x = torch.randn(2, 8, 8, dtype=torch.complex64, requires_grad=True)
unfold(x, (4, 4), (4, 4)).sum().backward()
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

## Where

`src/torchlinops/functional/_unfold/unfold.py`, `fold.py`, `src/torchlinops/functional/_interp/grid.py`, `ungrid.py` — the public wrappers reshape/complex-cast (`torch.view_as_complex` / `view_as_real`) and call kernel dispatchers; the output loses its grad fn. Note the adjoint functions exist (fold is the adjoint of unfold; ungrid/grid are adjoint pairs), so the math is available.

## Suggested fix

Two acceptable options; pick deliberately:

- implement forward+backward as a `torch.autograd.Function` wrapping the existing kernel calls (use the paired adjoint), or
- if non-differentiability is intended, document it and raise a clear error when inputs require grad (rather than a silent detached output).

Either way, add a regression test asserting `.backward()` produces the known adjoint (compare against `fold`/`grid` outputs).

## Value

Makes the functional kernels usable in training loops once compile support lands. This is a separate axis from compilation: the 720-cell compile matrix contains no eager execution failures for these kernels; the eager failures here are the four autograd-phase checks (one per functional kernel) and must not be read as compile regressions.
