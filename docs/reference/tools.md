# Tools

## `compile_check`

`torchlinops.tools.compile_check` profiles a user-provided linop with eager
execution versus `torch.compile` on a sample input. It always benchmarks the
wrapped operator (`linop`, `linop.H`, or `linop.N`), since that is what user code
calls in practice.

```python
import torch
from torchlinops import NUFFT
from torchlinops.tools import compile_check

locs = torch.randn(4096, 2, device="cuda")
A = NUFFT(locs, (64, 64), output_shape=("K",))
x = torch.randn(64, 64, dtype=torch.complex64, device="cuda")

result = compile_check(A, x, variants=["forward", "adjoint"])
```

The sample input determines the device, dtype, and input shape. An
output-shaped sample is constructed independently for adjoint profiling.
`variants` defaults to `["forward", "adjoint"]`; pass `"normal"` to include the
normal operator. The result includes:

- eager and compiled median times and speedup,
- numerical differences versus eager,
- graph-break count and locations (unless `fullgraph=True`),
- tensor shapes and byte sizes.

Autograd/backward profiling is not included yet. For full benchmark matrices,
see the [torch.compile benchmarks](../benchmarks/compile/index.md).

::: torchlinops.tools.compile_check
