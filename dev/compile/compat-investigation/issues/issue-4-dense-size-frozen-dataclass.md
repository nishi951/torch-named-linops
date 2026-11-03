# `Dense.size()` reconstructs frozen NamedDimension dataclasses during tracing

## Observed

CUDA and CPU `torch.compile` of `Concat` and `Stack` fail, while eager works:

```
torch._dynamo.exc.InternalTorchDynamoError:
NotImplementedError: currently can't reconstruct arbitrary frozen dataclass instances
```

Traceback threads through `NamedLinop.forward` → `check_input_shape` → `Dense.size` → `self.weightshape.index(dim)`:
`src/torchlinops/linops/dense.py:410`.

## Minimal repro

```python
import torch
from torchlinops import Dense, Concat

A = Dense(
    torch.randn(1, 3, 4, dtype=torch.complex64), ("N", "P", "Q"), ("N", "Q"), ("N", "P")
)
B = Dense(
    torch.randn(1, 3, 4, dtype=torch.complex64), ("N", "P", "Q"), ("N", "Q"), ("N", "P")
)
cat = Concat(A, B, idim="Q")
x = torch.randn(1, 8, dtype=torch.complex64)
torch.compile(lambda t: cat(t))(x)
# NotImplementedError: currently can't reconstruct arbitrary frozen dataclass instances
```

(`Stack` hits the same path.)

## Where

- `src/torchlinops/linops/dense.py:410` — `self.weight.shape[self.weightshape.index(dim)]`
- `src/torchlinops/linops/namedlinop.py:325` — `check_input_shape` calls `linop.size(dim)` in a generator during trace

## Suggested fix

Avoid list-search over frozen dataclasses inside traced code:

- precompute a plain `dict[str, int]` name→size map when building the Dense operator (`self._size_by_name`), and have `size(dim)` read from it;
- keep `weightshape` for eager/back-compat, but make the hot path integer/string-only.

Equivalent treatment for other linops whose `size()` walks named-dim tuples would prevent rediscovery of this class of bug.

## Value

Unblocks Concat/Stack under compile and any composed path that calls `check_input_shape` mid-graph. Small, local change.
