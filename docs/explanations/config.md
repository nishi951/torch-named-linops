# Configuration Options

`torchlinops` exposes a handful of global configuration flags in `torchlinops.config`:

```python
from torchlinops import config
import torchlinops.config as config  # equivalent
```

| Option | Default | Controls |
|---|---|---|
| `reduce_identity_in_normal` | `True` | Drop trivial `Identity` terms from `.N` results |
| `cache_adjoint_normal` | `True` | Cache `.H` and `.N` results on first access |
| `cache_nufft_parameters` | `True` | Cache internal NUFFT parameters |
| `log_device_transfers` | `True` | Log CUDA events, stream syncs, and device transfers |
| `shape_inference` | `False` | Resolve wildcards between shapes when building `Chain`s and `Add`s |

## Setting options

Assign directly for a persistent change:

```python
config.log_device_transfers = False
```

Or modify temporarily with the `config.using()` context manager, which restores every passed key on exit (and accepts multiple keys):

```python
with config.using(reduce_identity_in_normal=False, log_device_transfers=False):
    # both options are False here
    pass
# original values restored
```

Options are plain booleans. Most of them matter only in specific situations; the two you're most likely to reach for are described in detail below.

## `reduce_identity_in_normal`

Many operators satisfy $F^H F = I$ — for example, `FFT` with orthonormal normalization. When such an operator normal is folded into a chain, collapsing the resulting `Identity` produces a compact operator graph:

```python
import torch
from torchlinops import Dense, FFT, Dim, config

M, N = 3, 7
w = torch.randn(M, N)
D = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
F = FFT(ndim=1, grid_shapes=(Dim("M"), Dim("K")), batch_shape=Dim())
A = F @ D

print(A.N)
```

```
Dense((N,) -> (N1,))
```

Setting the flag to `False` keeps the full folded chain — useful for debugging or inspecting the exact computation:

```python
with config.using(reduce_identity_in_normal=False):
    A.reset_adjoint_and_normal()
    print(A.N)
```

```
Chain(
    Dense((N,) -> (M,))
    Identity((M,) -> (M,))
    Dense((M,) -> (N1,))
)
```

!!! warning "Clear the cache after changing this option"
    `.H` and `.N` results are cached on first access (see `cache_adjoint_normal` below). Call `linop.reset_adjoint_and_normal()` after toggling `reduce_identity_in_normal` (or use `config.using(...)` combined with the reset as above), or you will keep seeing operators built under the old setting.

## `shape_inference`

By default, shapes you leave unspecified with wildcards (`...`, `()`) stay as they are. With `shape_inference=True`, building a `Chain` or an `Add` resolves those wildcards against the shapes of the operators you compose with. For example, `Sampling` and `Diagonal` here have partially unspecified shapes:

```python
import torch
from torchlinops import FFT, Sampling, Diagonal, config
from torchlinops.linops.chain import Chain

C, K = 2, 7
idx = (torch.randint(0, 64, (K,)), torch.randint(0, 64, (K,)))
weights = torch.randn(C, K, dtype=torch.cfloat)

F = FFT(ndim=2, batch_shape=("C",))
S = Sampling(idx, input_size=(64, 64), batch_shape=("C",), output_shape=("K",))
D = Diagonal(weights)
```

Without shape inference (the default), both chains validate — since wildcards are permissive — but unresolved names linger in the prints:

```python
model = Chain(F, S, D)
print(model)
```

```
Chain(
    FFT((C, Nx, Ny) -> (C, Kx, Ky))
    Sampling((C, ...) -> (C, K))
    Diagonal((...,) -> (...,))
)
```

With shape inference enabled, each operator's input shape is resolved from the previous operator's output shape:

```python
from torchlinops import FFT, Sampling, Diagonal, config

F = FFT(ndim=2, batch_shape=("C",))
S = Sampling(idx, input_size=(64, 64), batch_shape=("C",), output_shape=("K",))
D = Diagonal(weights)

with config.using(shape_inference=True):
    model = Chain(F, S, D)
    print(model)
```

```
Chain(
    FFT((C, Nx, Ny) -> (C, Kx, Ky))
    Sampling((C, Kx, Ky) -> (C, K))
    Diagonal((C, K) -> (C, K))
)
```

A common pattern is a Tikhonov regularized normal operator, $A^H A + \lambda I$, with the `Identity`'s shape left unspecified:

```python
from torchlinops import Identity, config

with config.using(shape_inference=True):
    normal = model.N + 0.01 * Identity()
    print(normal)
```

```
Chain(
    FFT((C, Nx, Ny) -> (C, Kx, Ky))
    Sampling((C, Kx, Ky) -> (C, K))
    Diagonal((C, K) -> (C, K))
    Sampling.H((C, K) -> (C, Kx, Ky))
    FFT.H((C, Kx, Ky) -> (C, Nx, Ny))
) + Chain(
    Identity((C, Nx, Ny) -> (...,))
    Scalar((C, Nx, Ny) -> (C, Nx, Ny))
)
```

Here the `Add` picked the consensus shape `(C, Nx, Ny)` from `model.N` and the `Identity`'s dims were resolved against it.

Things to know about shape inference:

- **Opt-in**: the default is `False`.
- **Scope**: only `Chain.__init__` and `Add.__init__` infer; resolution runs *forward* through a chain (each operator's ishape is matched against the previous operator's oshape, never backwards).
- **Consensus**: for `Add`, shapes of all operands vote via `max_shape` — concrete dims beat wildcards, and for same-name dims the one with the larger index wins (`N1` beats `N`).
- **Validation still applies**: incompatible shapes raise errors exactly as before.

The mechanics of wildcard matching and resolution are described in [How Named Dimensions Propagate](named_dims_propagation.md).

## `cache_adjoint_normal`

By default, the first access to `.H` or `.N` builds the derived operator and caches it; subsequent accesses return the cached result. Because renamed dimensions do not automatically invalidate the cache, disabling the cache can be useful when shapes or config flags change during a session:

```python
import torch
from torchlinops import Dense, Dim, config

w = torch.randn(3, 7)
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))

_ = A.H  # adjoint cached here
A.oshape = ("Q",)  # renames M -> Q
print(A.H)  # stale: cached from before the rename
A.reset_adjoint_and_normal()
print(A.H)  # rebuilt from the current shape
```

```
Dense((M,) -> (N,))
Dense((Q,) -> (N,))
```

Use `linop.reset_adjoint_and_normal()` for spot invalidation, or `config.using(cache_adjoint_normal=False)`/`config.cache_adjoint_normal = False` to disable caching entirely (derived operators are rebuilt on every access).

## `cache_nufft_parameters`

`NUFFT` linops cache internal parameter tensors the first time they are called. This speeds up repeated applications but holds on to a few extra tensors; set `config.cache_nufft_parameters = False` to disable caching (e.g. under tight memory pressure or when NUFFT geometry changes between calls).

## `log_device_transfers`

When operators run across devices, `ToDevice` wrappers move tiles between them. With this flag enabled, low-level details — CUDA event creation, stream synchronization, device transfers — are logged on the `torchlinops` logger (via the `logging` module). Enable a handler to see the messages:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

Set `config.log_device_transfers = False` to silence this logging (or use `config.using(log_device_transfers=False)` for a temporary window).
