# Splitting and Batching Linops

Large linear operators can be too big to fit in memory, or too big to process in a single pass. Because dimensions are named, any linop can be split into tiles along its named dimensions, applied to the corresponding slices of the input, and reassembled — without changing the operator's semantics.

The modern API for this is `TilingStrategy`: a declarative description of *how* to tile (along which dimensions, with which tile edges), *where* to compute (per-tile device roles), and *from/to where* the data flows. This page shows the building blocks on CPU; for the full treatment of multi-GPU execution (streams, synchronization, memory mapping), see [Multi-GPU Execution](../explanations/multi_gpu.md).

## Tiling with `TilingStrategy`

A `TilingStrategy` describes the tiling with **bounds** — a mapping from dimension names to the *edges* of each tile along that dimension. Bounds don't have to be uniform, but they must span the full dimension:

```python
import torch
from torchlinops import Dense, Dim
from torchlinops.linops.split import TilingStrategy

CPU = torch.device("cpu")

w = torch.randn(3, 8)  # (M=3, N=8)
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
x = torch.randn(8)

strategy = TilingStrategy(
    bounds={"N": [0, 4, 8]},  # two tiles: N = [0:4], [4:8]
    devices=[CPU],
)
strategy.place_all(CPU)  # every tile: inputs, compute, and outputs on CPU
A_batched = strategy.apply(A)

print(A_batched)
print(torch.allclose(A_batched(x), A(x)))
```

```
Concat((N,) -> (M,)
    Dense((N,) -> (M,))
    Dense((N,) -> (M,))
    idim = N, odim = None
)
True
```

`strategy.apply(linop)` runs the pipeline **split → wrap → schedule**:

1. **split** — slice the operator into one tile per entry of the tile grid. Tiles keep the same dimension *names* as the parent; only the concrete sizes differ.
2. **wrap** — insert `ToDevice` transfers per tile: data moves from the tile's *input* device to its *compute* device, and results from there to its *output* device.
3. **schedule** — reassemble the tiles with `Concat`/`Add` into a single linop that behaves like the original (including `.H`, as verified above).

Because bounds are explicit edges, ragged tilings are fine: `bounds={"N": [0, 3, 8]}` makes tiles of sizes 3 and 5. The first edge starts coverage, and the last edge may fall short of the dimension's size — in that case the resulting operator is *truncated* on that dim: input-side dims act on the covered (first) coordinates of the input, and output-side dims produce only the covered slice of the output. Edges that overshoot the dimension's size raise a `ValueError` at split time, so coverage is never silently clamped:

```python
strategy = TilingStrategy(bounds={"M": [0, 3, 9]}, devices=[CPU])
strategy.place_all(CPU)
try:
    strategy.apply(A)
except ValueError as e:
    print(e)
```

```
TilingStrategy bounds on dim M exceed its size: last edge 9 > 3
```

(A tiling that covers less than the full dim is a *different operator* than the parent — that's the intent — but it still behaves like a name-preserving slice of it.)

!!! warning "Partial bounds create a different operator than the parent"
    If the last edge of a bound falls short of the dimension's size — or starts after `0` — the resulting operator is *different* from the parent: input-side dims act on the covered (first) coordinates of the input, and output-side dims return only the covered slice. The composite is a valid, internally consistent operator, but it prints with the same dimension *names* as the parent, so check the warning for what was actually covered: splitting with a partial bound emits a `UserWarning` naming the dim and the covered range (`edges cover [0, 3] less than [0, 8] -- this truncates the linop.`). Fully-covered tilings warn nothing; partially-covered ones are intentionally allowed, and the warning is your confirmation (or your cue to fix the edges if you expected full coverage).

### Reassembly semantics

Which container `schedule` picks depends on where the tiled dimension lives in the operator:

| Tiled dim appears in... | Behavior |
|---|---|
| input only (e.g. `N` of `Dense`) | each tile consumes its input slice; partial outputs are summed (`Concat` with no output dim) |
| output only (e.g. `M` of `Dense`) | each tile produces a disjoint slice of the output; outputs are concatenated |
| both (e.g. a batch dim) | inputs sliced per tile, outputs concatenated |

## Scheduling across devices

Each tile has three device roles — `DeviceRole.IN` (where its input data lives), `DeviceRole.COMPUTE` (where it runs), and `DeviceRole.OUT` (where results land). The placement helpers assign them fluently:

- `strategy.place_all(device)` — all tiles, all roles on one device.
- `strategy.distribute_across_devices(axis, roles=...)` — spread tiles of a single axis round-robin across the device list.
- `strategy.copy_role(src, dst)` — copy one role's assignment to another (e.g. outputs follow inputs so results land back on the base device).

Queries let you inspect the schedule before running anything: `device_of(tile_index, role)` reports a tile's device, and `transfers(src_role, dst_role)` lists exactly the data movements that would happen:

```python
import torch
from torchlinops.linops.split import TilingStrategy, DeviceRole

CPU = torch.device("cpu")
GPU = torch.device("cuda:0")

strategy = TilingStrategy(bounds={"N": [0, 4, 8]}, devices=[CPU, GPU])
(
    strategy.place_all(CPU)  # everything starts on CPU
    .distribute_across_devices(
        "N", roles=(DeviceRole.COMPUTE,)
    )  # ...compute round-robins over CPU/GPU
    .copy_role(
        DeviceRole.IN, DeviceRole.OUT
    )  # results return to where inputs came from
)
print([strategy.device_of((i,), DeviceRole.COMPUTE) for i in range(2)])
for tile, src, dst in strategy.transfers(DeviceRole.IN, DeviceRole.COMPUTE):
    print(f"tile {tile[0]}: {src} -> {dst}")
```

```
[device(type='cpu'), device(type='cuda', index=0)]
tile 1: cpu -> cuda:0
```

At this point, `strategy.transfers(DeviceRole.COMPUTE, DeviceRole.OUT)` is empty and the schedule is final. The last step — `strategy.apply(A)` or `strategy.split(A)` — actually moves data, and requires every referenced device to exist (e.g. a real GPU for `cuda:0`):

```python
# Requires GPUs; behaves exactly like A while running its tiles round-robin
A_batched = strategy.apply(A)
```

## Multi-level batching: `create_batched_linop_v2`

`create_batched_linop_v2(linop, [strategy1, strategy2, ...])` applies *several* strategies in order, recursively: the first strategy tiles the operator, then each remaining strategy is applied to every tile it produced. A typical use is splitting across two different axes at different levels (e.g. one strategy for device distribution, another for memory limits *within* each device):

```python
import torch
from torchlinops import Dense, Dim
from torchlinops.linops.split import TilingStrategy, create_batched_linop_v2

CPU = torch.device("cpu")

w = torch.randn(3, 8)
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
x = torch.randn(8)

s1 = TilingStrategy({"N": [0, 4, 8]}, [CPU])
s2 = TilingStrategy({"M": [0, 1, 3]}, [CPU])
s1.place_all(CPU)
s2.place_all(CPU)

B = create_batched_linop_v2(A, [s1, s2])
y = torch.randn(3)
print(torch.allclose(B(x), A(x)), torch.allclose(B.H(y), A.H(y)))
```

```
True True
```

`apply` and a single-strategy `create_batched_linop_v2` are equivalent (`strategy.apply(A)` ≡ `create_batched_linop_v2(A, [strategy])`).

### Options

`schedule` (and thus `apply` and `create_batched_linop_v2`) accepts options for how tiles execute:

| Option | Default | Behavior |
|---|---|---|
| `threaded` | `False` | Run tiles in parallel threads instead of sequentially |
| `num_workers` | `None` | Concurrency limit for thread pools |
| `accumulate` | `False` | Accumulate output batches; batch size is `num_workers` (threaded) or 1 (sequential) |

## Legacy APIs

The following APIs still work and underpin `TilingStrategy` internally, but new code should prefer the declarative `TilingStrategy` / `create_batched_linop_v2` API above.

### `split_linop`: manual tiling

`split_linop` returns `(linops, input_batches, output_batches)` — an array of tiles plus the input/output slices for each tile — and leaves reassembly to you:

```python
import torch
from torchlinops import Dense, Dim, split_linop

w = torch.randn(3, 8)
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
x = torch.randn(8)

linops, input_batches, output_batches = split_linop(A, {"N": 4})
print(linops.shape)  # (2,) — two tiles
print([sl[0] for sl in input_batches])  # [slice(0, 4, None), slice(4, 8, None)]

ys = [t(x[sl[0]]) for t, sl in zip(linops, input_batches)]
y = torch.stack(ys).sum(dim=0)  # N is an input-only dim: partial outputs must be summed
print(torch.allclose(y, A(x)))
```

```
(2,)
[slice(0, 4, None), slice(4, 8, None)]
True
```

### `create_batched_linop` + `BatchSpec`

`create_batched_linop` (with `BatchSpec` chunk *sizes* rather than explicit edges) wraps splitting, reassembly, and device placement according to the spec:

```python
from torchlinops.linops.split import create_batched_linop, BatchSpec

spec = BatchSpec(
    batch_sizes={"N": 4},
    device_matrix=[torch.device("cpu")],
    base_device=torch.device("cpu"),
)
A_batched = create_batched_linop(A, spec)
print(torch.allclose(A_batched(x), A(x)))
```

```
True
```

Each operator only needs to implement `split` to describe how it decomposes along its dimensions (the same hook `TilingStrategy` uses). Not every dimension is splittable for every operator — each operator declares the dimensions it can split along (e.g. `Pad` cannot split its padding dims). Device placement and reassembly details live in [Multi-GPU Execution](../explanations/multi_gpu.md).
