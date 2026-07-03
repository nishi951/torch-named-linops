# Benchmark Methodology

## Overview

The benchmarking suite measures the performance of core `torch-named-linops`
operations against [SigPy](https://github.com/mikgroup/sigpy), a widely used
scientific computing package. Three variants are compared per operation:

| Variant | Description |
|---------|-------------|
| **torchlinops (functional)** | Direct calls to `torchlinops.functional` functions (e.g. `nufft`, `interpolate`, `array_to_blocks`). No linop abstraction overhead. |
| **torchlinops (linop)** | Calls through the `NamedLinop` abstraction (e.g. `A(x)`, `A.H(y)`). Includes dimension tracking and dispatch overhead. |
| **sigpy** | Equivalent operations in SigPy (e.g. `sp.nufft`, `sp.interp.interpolate`, `sp.array_to_blocks`). |

## Data Generation

Each benchmarked function **generates fresh input data on every iteration**.
This prevents GPU L2 cache effects that would artificially speed up repeated
computation on the same tensor. For small inputs (e.g. a 64×64 grid ≈ 32 KB),
the data fits entirely in the RTX 3090's 6 MB L2 cache, so caching would
otherwise dominate the measurement.

The timed function includes:

1. Random input generation (`torch.randn` / `xp.random.randn`)
2. Random location generation (`_get_valid_locs`)
3. The actual operation

For the linop variant, the linop is **constructed once outside the timed loop**
so that construction cost (apodization weights, padding parameters, etc.) is
not included in the benchmark.

### Data Generation Reporting

A **separate data-generation-only benchmark** is run for each variant using
the same `blocked_autorange` methodology. The mean data generation time is
reported in the "Data Gen" column for reference, but is **not subtracted**
from the operation's total mean time. The "Mean" column reports the total
time including data generation.

What each variant's data-gen function includes:

| Variant | Data gen includes |
|---------|-------------------|
| torchlinops (functional) | `torch.randn` + `_get_valid_locs` |
| torchlinops (linop) | `torch.randn` |
| sigpy | `xp.random.randn` + `_get_valid_locs` + `from_pytorch` (+ `with dev:` for GPU) |

## Timing

### CPU and torchlinops GPU

Timing uses `torch.utils.benchmark.Timer.blocked_autorange()` with:

- `min_run_time = 0.2` seconds for CPU benchmarks
- `min_run_time = 0.05` seconds for GPU benchmarks

`blocked_autorange` first runs an auto-calibration phase (doubling the number
of runs per block until the block time exceeds measurement overhead), which
also serves as warm-up. It then collects measurements until the target total
time is reached, providing statistically robust mean, median, and IQR.

### sigpy GPU

SigPy operations on GPU use CuPy arrays rather than PyTorch tensors. Timing
uses a custom `cupy_blocked_autorange()` function that mirrors
`torch.utils.benchmark.Timer.blocked_autorange()` but uses:

- `cp.cuda.Event` for GPU timing (with `event.synchronize()`)
- `cp.cuda.runtime.deviceSynchronize()` for warm-up synchronization
- `cp.get_default_memory_pool().total_bytes()` for peak memory tracking

The auto-calibration and collection phases follow the same algorithm as
`blocked_autorange`.

## Memory Tracking

| Device | Method |
|--------|--------|
| GPU (torchlinops) | `torch.cuda.max_memory_allocated()` |
| GPU (sigpy) | `cp.get_default_memory_pool().total_bytes()` |
| CPU | Not tracked (reported as "—") |

## Operations Benchmarked

### NUFFT

Non-uniform FFT forward and adjoint, 2D and 3D.

- **torchlinops functional**: `nufft(x, locs, oversamp, width)` / `nufft_adjoint(y, locs, grid_size, oversamp, width)`
- **torchlinops linop**: `NUFFT(locs, grid_size, ...)` then `A(x)` / `A.H(y)`
- **sigpy**: `sp.nufft(x, coord, oversamp, width)` / `sp.nufft_adjoint(y, coord, grid_size, oversamp, width)`

### Interpolate

Kaiser-Bessel interpolation forward (grid → points) and adjoint (points → grid), 2D and 3D.

- **torchlinops functional**: `interpolate(x, locs, width, ...)` / `interpolate_adjoint(y, locs, grid_size, width, ...)`
- **torchlinops linop**: `Interpolate(locs, grid_size, ...)` then `A(x)` / `A.H(y)`
- **sigpy**: `sp.interp.interpolate(x, coord, ...)` / `sp.interp.gridding(y, coord, grid_size, ...)`

### ArrayToBlocks / BlocksToArray

Sliding window block extraction (forward) and accumulation (adjoint), 3D.

- **torchlinops functional**: `array_to_blocks(x, block_size, stride)` / `blocks_to_array(x, grid_size, block_size, stride)`
- **torchlinops linop**: `ArrayToBlocks(...)` / `BlocksToArray(...)` then `A(x)`
- **sigpy**: `sp.array_to_blocks(x, block_size, stride)` / `sp.blocks_to_array(x, grid_size, block_size, stride)`

## Problem Sizes

Each operation is benchmarked at three problem sizes (small, medium, large)
to show how performance and memory scale with input size.

### NUFFT 2D / Interpolate 2D

| Size | Grid | npts | Problem size |
|------|------|------|-------------|
| Small | 64×64 | 4,096 | 4,096 |
| Medium | 128×128 | 16,384 | 16,384 |
| Large | 256×256 | 65,536 | 65,536 |

The number of points equals `prod(grid_size)` (fully sampled).

### NUFFT 3D / Interpolate 3D

| Size | Grid | npts | Problem size |
|------|------|------|-------------|
| Small | 32³ | 4,096 | 32,768 |
| Medium | 64³ | 32,768 | 262,144 |
| Large | 128³ | 262,144 | 2,097,152 |

The number of points is 1/8 of `prod(grid_size)` (realistic 3D undersampling).

### ArrayToBlocks 3D

| Size | Grid | Block | Stride | Problem size |
|------|------|-------|--------|-------------|
| Small | 32³ | 8³ | 4 | 32,768 |
| Medium | 64³ | 8³ | 4 | 262,144 |
| Large | 128³ | 8³ | 4 | 2,097,152 |

The "problem size" is `prod(grid_size)`. The large size (128³) produces
~15M output elements (~120 MB for complex64).

## Charts

### Bar Charts

One grouped bar chart per problem size per device, showing mean time for all
operations × libraries. Bars are grouped by library:

- **torchlinops** (blue): functional API
- **torchlinops (linop)** (light blue): NamedLinop abstraction
- **sigpy** (orange): SigPy reference

### Scaling Curves

Scaling curve figures per device:

- **Timing scaling**: One subplot per operation type (NUFFT 2D,
  NUFFT 3D, Interpolate 2D, Interpolate 3D, ArrayToBlocks 3D). X-axis is
  problem size (log scale), Y-axis is mean time (log scale). Lines are per
  library × direction (solid = forward, dashed = adjoint).

- **Memory scaling**: One subplot per operation type (GPU only). X-axis is
  problem size, Y-axis is peak GPU memory (log scale). Same line/marker
  conventions as timing.

## Running Benchmarks

```bash
# CPU benchmarks only
just bench-cpu

# All benchmarks (CPU + GPU, requires CUDA for GPU tests)
just bench-gpu
# or equivalently:
just bench

# Regenerate docs from latest results
just bench-report

# Full docs build (includes benchmark report)
just docs
```

Results are saved to `benchmarks/results/latest/` and automatically archived
to `benchmarks/results/<date>-<sha>/`. If the working tree is dirty, a
`patch.diff` is saved alongside the results for reproducibility.
