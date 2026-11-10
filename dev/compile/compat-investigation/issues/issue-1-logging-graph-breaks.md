# Guard CUDA logging inside the NamedLinop wrapper (graph breaks + fullgraph failure)

## Observed

- `torch.compile` default mode: 29 cells break the graph at `logging.Logger method not supported for non-export cases`; the two graph breaks in the simple 20-step pipeline come from here too.
- `torch.compile(fullgraph=True)`: hard failure for Add, Chain, Repeat, Sampling, SamplingNUFFT, FastSamplingNUFFT, SumReduce, Truncate (9 operators) and the simple pipeline.

## Minimal repro

```python
import torch
from torchlinops import Dense, Chain

W1 = torch.randn(8, 16, device="cuda")
W2 = torch.randn(8, 8, device="cuda")
chain = Chain(
    Dense(W1, ("M", "N"), ("N",), ("M",)),
    Dense(W2, ("M", "M"), ("M",), ("M",)),
)
x = torch.randn(16, device="cuda")
torch.compile(lambda t: chain(t), fullgraph=True)(x)
# torch._dynamo.exc.Unsupported: logging.Logger method not supported for non-export cases
```

Default mode succeeds but reports graph breaks.

## Where

- `src/torchlinops/linops/namedlinop.py:29-31` (`_log_transfer`, guarded only by `config.log_device_transfers`)
- `src/torchlinops/linops/namedlinop.py:711` and `:719` (`SyncContext.__post_init__`, unconditional `logger.debug` on the CUDA path)
- `src/torchlinops/linops/namedlinop.py:349` (additional `logger.debug`)

## Suggested fix

Skip logging while compiling, e.g. in `_log_transfer` and `SyncContext.__post_init__`:

```python
if torch.compiler.is_compiling():
    return
```

or hoist the level check (`if logger.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():`). Keep eager logging unchanged.

## Value

Removes every observed wrapper logging break on CUDA, makes `fullgraph=True` viable for the affected operators once other blockers are fixed, and removes the 2 breaks/20-step pipeline.

## Notes

`Sleep` additionally fails fullgraph on `torch.cuda.stream(...)`; that is a separate limitation, not covered by this issue.
