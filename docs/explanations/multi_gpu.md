# Multi-GPU Execution

`torchlinops` can distribute a linear operator across multiple GPUs. This is useful when a single GPU does not have enough memory for the full operator, or when independent tiles can run in parallel for faster execution.

This page covers both halves of multi-GPU execution:

1. **Splitting and placement** — how a linop is tiled, moved to target devices, and reassembled.
2. **Synchronization** — how `SyncContext`, CUDA events, and streams ensure that parallel tiles do not read data before it is ready.

We assume CUDA devices with peer-to-peer memory access.

## The splitting mechanism

At its core, multi-GPU distribution is built on the ability to **split** a linop into smaller sub-linops that each operate on a slice of the data.

### `split`

Every `NamedLinop` implements `split(linop, tile)`, where `tile` is a dictionary mapping dimension names to slices. The method returns a new linop that operates only on the specified slice.

For example, a `Diagonal` linop with shape `(Nx, Ny) -> (Nx, Ny)` and a weight tensor of shape `(256, 256)` can be split along `Nx` into two sub-linops, each with a weight of shape `(128, 256)`.

```python
from torchlinops.nameddim import NamedDimension as ND

tile = {ND("Nx"): slice(0, 128)}
sub_linop = NamedLinop.split(A, tile)
```

For adjoint splitting, use `adj_split(linop, tile)` which constructs the adjoint, splits it according to *tile*, and returns the adjoint of the split.

### `split_linop`

The `split_linop()` function automates the tiling process. Given a linop and a dictionary of batch sizes, it:

1. Queries `linop.size(dim)` for every dimension to determine the total size.
2. Creates a grid of tiles, where each tile maps dimensions to `(index, slice)` pairs.
3. Calls `split()` for each tile, producing an nd-array of sub-linops.

```python
from torchlinops.linops.split import split_linop

# Split a linop into chunks of 128 along Nx and 64 along Ny
linops, ibatches, obatches = split_linop(A, {"Nx": 128, "Ny": 64})
# linops is a 2D numpy array of sub-linops
```

### Chain splitting

When splitting a `Chain` (a composition of linops), each constituent linop is split independently according to the slices over its own dimensions. This means that a chain $C \circ B \circ A$ is split into tiles where each tile is a chain of the corresponding sub-linops.

The `Chain.split()` method distributes the tile dictionary to each constituent linop based on dimension names.

## BatchSpec

`BatchSpec` is a dataclass that bundles all the information needed to split and distribute a linop:

| Field | Description |
|-------|-------------|
| `batch_sizes` | `dict[dim, int]` -- how large each chunk should be |
| `device_matrix` | Optional array of `torch.device` objects, one per tile |
| `base_device` | The device where input/output tensors live |

The `device_matrix` is broadcast to match the tile grid shape. For example, if splitting creates a 4-tile grid and `device_matrix = ["cuda:0", "cuda:1"]`, it is repeated to `["cuda:0", "cuda:1", "cuda:0", "cuda:1"]`. The broadcasting uses a fuzzy strategy that tiles and truncates as needed, so the device list does not need to exactly match the number of tiles.

`BatchSpec` has a `broadcast_device_matrix(linop)` method that computes the number of tiles along each batched dimension and broadcasts the device matrix accordingly.

## `create_batched_linop`

This is the main entry point for multi-GPU distribution. It takes a linop and one or more `BatchSpec` objects and returns a new composite linop that transparently handles splitting, device placement, and reassembly.

### How it works

1. **Split** the linop into tiles according to `batch_sizes`.
2. **Place** each tile on its target device using `ModuleMemoryMap.memory_aware_to()`, which preserves tensor storage topology (see [Copying Linops](copying_linops.md)).
3. **Wrap** each tile with `ToDevice` linops for input transfer (base -> target) and output collection (target -> base).
4. **Reassemble** tiles by reducing along each split dimension:
    - If the dimension appears in both `ishape` and `oshape`: use `Concat` (the tiles partition the data along that dim).
    - If the dimension appears only in `ishape` or only in `oshape`: use `Concat` along the relevant side.
    - If the dimension appears in neither: use `Add` (the tiles produce partial results that must be summed).

The result is a single composite linop that behaves identically to the original but executes across multiple devices.

### Recursive batching

`create_batched_linop` accepts a *list* of `BatchSpec` objects and processes them recursively. This enables multi-level splitting -- for example, first splitting across GPUs along one dimension, then splitting within each GPU along another dimension for memory management.

## Data transfer

### DeviceSpec

`DeviceSpec` is a lightweight dataclass that holds useful CUDA-related objects for multi-GPU computation:

| Field | Description |
|-------|-------------|
| `device` | The `torch.device` for this specification |
| `compute_stream` | Stream used for computation on this device |
| `transfer_stream` | Stream used for data transfers from this device |

`DeviceSpec` has a `p2p_setup(other_device)` method that configures compute and transfer streams for peer-to-peer transfers between devices. This is called automatically when creating `ToDevice` linops between CUDA devices. For a CUDA device, `p2p_setup` lazily creates:

- `compute_stream = default_stream(device)`
- `transfer_stream = Stream(device)` (a new dedicated stream)

There is no global registry of transfer streams; each `DeviceSpec` owns its own `transfer_stream`.

### `ToDevice`

`ToDevice` is a specialized linop that moves tensors between devices. It is the glue between the base device (where input/output data lives) and the target devices (where computation happens).

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `ispec` | `DeviceSpec` | Source (input) device specification |
| `ospec` | `DeviceSpec` | Target (output) device specification |
| `is_gpu2gpu` | `bool` | True if both source and target are CUDA devices |

The adjoint of `ToDevice(A -> B)` is `ToDevice(B -> A)` -- it simply reverses the direction and swaps the device specs.

For CUDA-to-CUDA transfers, `ToDevice` uses non-blocking operations on dedicated streams:

```
Input on base_device
  -> transfer_stream: non-blocking .to(target_device)
  -> target_stream: wait for transfer, run computation
  -> base_stream: collect output back to base_device
```

Key implementation details:

- `x.record_stream(transfer_stream)` prevents PyTorch's caching allocator from freeing the source tensor's memory before the transfer completes.
- `target_stream.wait_stream(transfer_stream)` ensures the target stream does not start computation until the data has arrived.
- The transfer stream is the `ispec.transfer_stream` attribute set by `p2p_setup`.

## Synchronization with `SyncContext`

Splitting a linop across GPUs is only half the story. The other half is making sure that asynchronous transfers do not outrun the producer of the data.

### The problem

Suppose you build a multi-GPU linop with a CUDA base device:

```python
base = torch.device("cuda:0")
gpu1 = torch.device("cuda:1")

OnDevice = Add(
    Chain(ToDevice(base, base), A0),
    Chain(ToDevice(base, gpu1), A1),
    threaded=True,
)
```

`Add` runs its two children in parallel threads. The second child transfers the input tensor from `cuda:0` to `cuda:1` on a dedicated transfer stream. If the input tensor is still being produced by earlier GPU work on `cuda:0`'s default stream, that transfer must wait until the work is done. Without a barrier, the transfer could copy partial or stale data.

`SyncContext` is the lightweight object that carries that barrier.

### The `SyncContext` dataclass

`SyncContext` lives in `torchlinops.linops.namedlinop`. It is created fresh on every `forward()` call and carries four fields:

```python
@dataclass
class SyncContext:
    linop: type[NamedLinop]        # Class of the linop that owns this context
    input_device: torch.device     # Device of the input tensor
    parent: Optional[SyncContext]  # Calling context (None at the top level)
    start_event: torch.cuda.Event | None  # Barrier event for CUDA sync
```

`start_event` is the important field. It marks "everything before this point is done" on the input device's stream.

### How events are created

`SyncContext.__post_init__` decides whether to create, reuse, or skip the event:

1. **CPU input** (`input_device.type != "cuda"`) → no event. Synchronization is a no-op on CPU.
2. **Reuse parent's event** → if `parent` is a container (`is_container=True`) and `parent.start_event` exists, reuse it. This means: "I am a child of a parallel container; my siblings and I all share the same starting barrier."
3. **Record a fresh event** → if no event was reused and the current linop is a container (`is_container=True`), record a fresh event on the current stream of `input_device`. This means: "I am a parallel container; I am the barrier for my children."
4. **No event** → otherwise. This happens for non-container children of non-containers (sequential execution on the same stream).

Only four linops are containers: `Add`, `Concat`, `Stack`, and `Chain`. They set `is_container = True`.

### Opt-in via signature inspection

Not every linop participates in synchronization. `NamedLinop.forward()` inspects the signature of `self.fn` at runtime:

```python
if "context" in inspect.signature(self.fn).parameters:
    context = SyncContext(...)
    return self._run(x, context)
return self._run(x)
```

Only linops whose `fn` declares a `context` parameter receive a `SyncContext`. This keeps simple linops (`Dense`, `Diagonal`, `FFT`, `Identity`, ...) completely free of sync overhead.

| Linop | Declares `context`? | `is_container`? | Role |
|-------|---------------------|-----------------|------|
| `Add` | yes | yes | records barrier, shares it with children |
| `Concat` | yes | yes | records barrier, shares it with children |
| `Stack` | yes | yes | records barrier, shares it with children |
| `Chain` | yes | yes | records barrier, forwards to first child only |
| `ToDevice` | yes | no | **consumer** — waits on `start_event` |
| `Dense`, `Diagonal`, `FFT`, `Identity`, ... | no | no | unaffected, zero overhead |

### How `parallel_execute` shares one event

`Add`, `Concat`, and `Stack` run their children through `parallel_execute()` from `torchlinops.linops.schedule`. The key detail is that every child is called with the **same** container context:

```python
return parallel_execute(
    linops,
    inputs,
    context,      # the container's own SyncContext
    reduce_fn=...,
    threaded=...,
    num_workers=...,
)
```

Each child's `forward()` then builds a new `SyncContext` with `parent=context`. Because the parent is a container with a `start_event`, the child **reuses** that event. The result is *N* children waiting on **one** barrier, rather than each child recording its own event.

When `threaded=True`, the children run in a `ThreadPoolExecutor`; when `threaded=False`, they run sequentially. In both cases the same context is passed, so synchronization is correct either way.

### How `Chain` propagates context

`Chain` is a container, but it runs its children sequentially. It therefore forwards the context to **only the first child**:

```python
def fn(chain, x, context=None):
    x = chain[0](x, context)          # first child inherits the barrier
    for linop in chain.linops[1:]:
        x = linop(x)                  # remaining children run sequentially
    return x
```

This is correct for the usual multi-GPU pattern `Chain(ToDevice(base, gpu), A, ToDevice(gpu, base))`:

- The **leading** `ToDevice` needs the barrier so it does not transfer input before it is ready.
- `A` runs sequentially after the transfer, so it needs no explicit sync.
- The **trailing** `ToDevice` also runs sequentially after `A`; it creates a fresh context with `parent=None` and falls back to waiting on the current stream of its input GPU, which already includes `A`'s work.

### `ToDevice` as the event consumer

`ToDevice.fn` is the only place `start_event` is actually consumed:

```python
def fn(todevice, x, context):
    return todevice._fn(
        x,
        todevice.ispec,
        todevice.ospec,
        wait_for_event=context.start_event,
    )
```

For GPU-to-GPU transfers, `_gpu2gpu_transfer` does:

```python
if wait_for_event is not None:
    transfer_stream.wait_event(wait_for_event)
else:
    transfer_stream.wait_stream(current_stream(x.device))

with torch.cuda.stream(transfer_stream):
    out = x.to(odevice, non_blocking=True)
    x.record_stream(transfer_stream)

target_stream.wait_stream(transfer_stream)
```

If `context.start_event` is present, the transfer stream waits on that event. If it is absent (for example, a `ToDevice` called directly without a container), the transfer stream waits on the current stream of the source device instead.

### A complete worked example

The following pattern is the heart of multi-GPU execution in `torchlinops` (adapted from the test suite). The base device is `cuda:0`; one tile runs locally on `cuda:0`, the other is transferred to `cuda:1`:

```python
import torch
from torchlinops import Add, Chain, Dense, Dim, Sleep, ToDevice
from torchlinops.nameddim import NamedDimension as ND

base = torch.device("cuda:0")
gpu1 = torch.device("cuda:1")

N = 8192

def slow_linop(N: int, sleep_duration: float = 0.1):
    weight = torch.randn(N, N)
    dense = Dense(weight, Dim("MN"), Dim("N"), Dim("M"))
    sleep = Sleep(sleep_duration, ioshape=(ND("M"),))
    return sleep @ dense

A0 = slow_linop(N)
A1 = slow_linop(N)

OnDevice = Add(
    Chain(
        ToDevice(base, base, ioshape=A0.ishape),
        A0.to(base),
        ToDevice(base, base, ioshape=A0.oshape),
    ),
    Chain(
        ToDevice(base, gpu1, ioshape=A1.ishape),
        A1.to(gpu1),
        ToDevice(gpu1, base, ioshape=A1.oshape),
    ),
    threaded=True,
)

x = torch.randn(N, device=base)
y = OnDevice(x)  # runs A0 and A1 in parallel on two GPUs
```

Here is the event trace for the CUDA synchronization:

1. `OnDevice(x)` → `Add.forward(x, context=None)`.
   - `Add.fn` accepts `context`, so `forward()` creates `SyncContext(linop=Add, input_device=base, parent=None)`.
   - `Add` is a container with CUDA input and no parent event, so it **records a fresh event** `E0` on `base`'s current stream.
   - `E0` marks: "the input tensor `x` is ready on `cuda:0`."

2. `Add.fn` calls `parallel_execute([Chain0, Chain1], [x, x], AddCtx, threaded=True)`.
   - Two worker threads start.
   - Each worker calls `Chain_i(x, AddCtx)` with the **same** `AddCtx`.

3. **Worker for `Chain0` (local, `cuda:0`):**
   - `Chain0.forward(x, AddCtx)` creates `SyncContext(linop=Chain, input_device=base, parent=AddCtx)`.
   - Parent `Add` is a container with event `E0`, so `Chain0` **reuses `E0`**.
   - `Chain0.fn` calls `Chain0[0](x, Chain0Ctx)` = `ToDevice(base, base)`.
   - Same device → `_gpu2gpu_transfer` returns `x` unchanged (no transfer needed).
   - `A0` runs on `base`'s default stream; the trailing `ToDevice(base, base)` is also a no-op.

4. **Worker for `Chain1` (remote, `cuda:1`):**
   - `Chain1.forward(x, AddCtx)` creates `SyncContext(linop=Chain, input_device=base, parent=AddCtx)`.
   - Parent `Add` is a container with `E0`, so `Chain1` **reuses `E0`**.
   - `Chain1.fn` calls `Chain1[0](x, Chain1Ctx)` = `ToDevice(base, gpu1)`.
   - `ToDevice.forward(x, Chain1Ctx)` creates `SyncContext(linop=ToDevice, input_device=base, parent=Chain1Ctx)`.
   - Parent `Chain` is a container with `E0`, so `ToDevice` **reuses `E0`**.
   - `ToDevice.fn` → `_gpu2gpu_transfer` → `transfer_stream.wait_event(E0)`. The transfer to `gpu1` cannot start until `E0` completes.
   - After the transfer, `A1` runs on `gpu1`'s default stream.
   - The trailing `ToDevice(gpu1, base)` transfers the result back to `base`.

**The key result:** both workers share the **same** event `E0`. As soon as `E0` completes, `A0` starts on `cuda:0` and the transfer to `cuda:1` begins concurrently. Once the transfer arrives, `A1` starts on `cuda:1`, overlapping with `A0`.

### Nested composites

Events propagate correctly through nested containers. Consider:

```python
Concat(
    Add(A, B),   # inner Add
    C,
    idim=('M',),
)(x)
```

1. `Concat` records event `E0`.
2. The inner `Add` receives `Concat`'s context, reuses `E0`, and passes `E0` to `A` and `B`.
3. `C` receives `Concat`'s context and reuses `E0`.

One barrier synchronizes the entire tree.

### Edge cases

- **CPU inputs**: `SyncContext.__post_init__` skips all event logic when `input_device.type != "cuda"`. The context is created, but `start_event` stays `None`, so there is no overhead.
- **Direct `ToDevice` calls**: calling `ToDevice(gpu0, gpu1)(x)` outside any container creates a context with `parent=None`. `ToDevice` is not a container, so no event is recorded; `wait_for_event=None`, and the transfer falls back to `transfer_stream.wait_stream(current_stream(x.device))`.
- **`threaded=False`**: `parallel_execute` still passes the same context to every child, so synchronization remains correct. Sequential mode is useful for debugging.

## Visualizing the sync graph

For debugging, you can log the logical synchronization graph with `torchlinops.cuda_trace.cuda_logger`:

```python
import torchlinops.config as config
from torchlinops.cuda_trace import cuda_logger

with config.using(log_cuda_events=True, log_device_transfers=True):
    y = OnDevice(x)
    print(cuda_logger.display(reset=True))
```

The output shows:

- `●` record — an event was recorded (marks completion of prior work).
- `◇` wait — a stream is waiting for an event or another stream.
- `○` implicit — an auto-created node such as the default stream at operation start.
- `←` blocks on — a node cannot start until the target completes.

Nodes are grouped into topological stages; operations in the same stage can run in parallel, and later stages are blocked on earlier ones. This is the easiest way to confirm that multi-GPU transfers really do wait on the shared barrier and that compute kernels overlap across devices.

## Base device and stream layout

The **base device** is the device on which the input is required and on which the final output is produced. It orchestrates all transfers. For a two-GPU setup with `cuda:0` as base, the default stream layout is:

- `cuda:0` (base)
  - default stream: computation on `cuda:0`
  - transfer stream: moving tensors between `cuda:0` and `cuda:1`
- `cuda:1`
  - default stream: computation on `cuda:1`

## Configuration

Two config flags are useful for debugging multi-GPU execution:

- `torchlinops.config.log_device_transfers` (default `True`): logs device-transfer messages from `ToDevice` and related utilities.
- `torchlinops.config.log_cuda_events` (default `False`): records CUDA `record_event` / `wait_event` / `wait_stream` calls as a labeled dependency graph for `cuda_logger.display()`.

```python
import torchlinops.config as config

config.log_device_transfers = True   # log transfers
config.log_cuda_events = True        # log event graph
```

## Limitations and future work

- **Peer-to-peer access**: The current implementation assumes efficient P2P memory access between GPUs. Systems without P2P will fall back to staging through host memory, which is slower.
- **Manual tuning**: Choosing optimal `batch_sizes` and `device_matrix` requires understanding the model's memory footprint and the hardware topology. No auto-tuning is provided.
- **Single-node only**: Running computations on distributed GPU nodes across multiple servers is possible in principle via standard PyTorch distributed APIs, but no simplified API is provided within this library.
- **Autograd and shared linops**: Parallel containers such as `Add(A, A)` use the same linop object in multiple threads. For inference this is safe, but backpropagating through such a construct can race on parameter gradients because PyTorch's gradient accumulation is not atomic across threads. If you need gradients from a threaded multi-GPU linop, give each tile its own copy of the weights (for example, via `create_batched_linop`, which places independent copies on each device).
