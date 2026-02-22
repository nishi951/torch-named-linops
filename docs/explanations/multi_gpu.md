# Multi-GPU Splitting

Linops should be able to take advantage of multi-GPU systems to leverage the larger total GPU memory available and to gain increased speed from parallelization across separate devices.

We assume CUDA devices with peer-to-peer memory access.

## The splitting mechanism

At its core, multi-GPU distribution is built on the ability to **split** a linop into smaller sub-linops that each operate on a slice of the data.

### `split_forward`

Every `NamedLinop` can implement `split_forward(ibatch, obatch)`, where `ibatch` and `obatch` are lists of slices corresponding to the input and output dimensions. The method returns a new linop that operates only on the specified slice.

For example, a `Diagonal` linop with shape `(Nx, Ny) -> (Nx, Ny)` and a weight tensor of shape `(256, 256)` can be split along `Nx` into two sub-linops, each with a weight of shape `(128, 256)`.

The static `split(linop, tile)` method provides a higher-level interface that accepts a dictionary mapping dimension names to slices:

```python
from torchlinops.nameddim import NamedDimension as ND

tile = {ND("Nx"): slice(0, 128)}
sub_linop = A.split(A, tile)
```

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

## BatchSpec

`BatchSpec` is a dataclass that bundles all the information needed to split and distribute a linop:

| Field | Description |
|-------|-------------|
| `batch_sizes` | `dict[dim, int]` -- how large each chunk should be |
| `device_matrix` | Optional array of `torch.device` objects, one per tile |
| `base_device` | The device where input/output tensors live |
| `base_stream` | CUDA stream for the base device |
| `transfer_stream` | CUDA stream used for data transfers |

The `device_matrix` is broadcast to match the tile grid shape. For example, if splitting creates a 4-tile grid and `device_matrix = ["cuda:0", "cuda:1"]`, it is repeated to `["cuda:0", "cuda:1", "cuda:0", "cuda:1"]`. The broadcasting uses a fuzzy strategy that tiles and truncates as needed, so the device list does not need to exactly match the number of tiles.

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

## Data transfer and synchronization

### `ToDevice`

`ToDevice` is a specialized linop that moves tensors between devices. It is the glue between the base device (where input/output data lives) and the target devices (where computation happens).

For CUDA-to-CUDA transfers, `ToDevice` uses non-blocking operations on specific streams:

```
Input on base_device
  -> transfer_stream: non-blocking .to(target_device)
  -> target_stream: wait for transfer, run computation
  -> base_stream: collect output back to base_device
```

Key implementation details:

- `x.record_stream(stream)` prevents PyTorch's caching allocator from freeing the source tensor's memory before the transfer completes.
- `ostream.wait_stream(istream)` ensures the target stream does not start computation until the data has arrived.
- The adjoint of `ToDevice(A -> B)` is `ToDevice(B -> A)` -- it simply reverses the direction and swaps the streams.

### `RepeatedEvent`

`RepeatedEvent` is a lightweight wrapper around CUDA events that creates a fresh event on each `record()` call. This is used as the `start_event` on the top-level batched linop: when `forward()` is called, it records an event that all `ToDevice` input transfers wait on. This ensures that all tiles start their transfers simultaneously, enabling maximum overlap between transfer and computation.

### Stream workflow

The full execution flow for a multi-GPU forward pass:

1. `start_event.record()` on the current stream -- signals that input data is ready.
2. For each tile:
    - `transfer_stream.wait_event(start_event)` -- wait for input.
    - Transfer input slice to target device via `transfer_stream`.
    - `target_stream.wait_stream(transfer_stream)` -- wait for data arrival.
    - Compute on `target_stream`.
    - Transfer output back to base device.
3. Reassemble outputs on base device (via `Concat` or `Add`).

## Limitations and future work

- **Peer-to-peer access**: The current implementation assumes efficient P2P memory access between GPUs. Systems without P2P will fall back to staging through host memory, which is slower.
- **Manual tuning**: Choosing optimal `batch_sizes` and `device_matrix` requires understanding the model's memory footprint and the hardware topology. No auto-tuning is provided.
- **Single-node only**: Running computations on distributed GPU nodes across multiple servers is possible in principle via standard PyTorch distributed APIs, but no simplified API is provided within this library.
