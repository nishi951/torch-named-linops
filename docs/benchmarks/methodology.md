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

The benchmark handlers (`TorchHandler` and `CupyHandler`) implement this via
the `data_gen_fn` parameter to `blocked_autorange()`:

```python
# Simplified pseudocode of handler's timing loop
for each iteration:
    data = data_gen_fn()    # NOT timed
    trial_start()
    fn(data)                # timed
    trial_end()
```

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

All benchmarks use a unified handler-based approach via `TorchHandler` and `CupyHandler`, which provide `blocked_autorange()` methods with automatic iteration count determination.

### torchlinops (CPU and GPU)

`TorchHandler.blocked_autorange()` handles both CPU and CUDA timing:

- **GPU**: Uses `torch.cuda.Event` for accurate GPU timing with `event.synchronize()`
- **CPU**: Uses `time.perf_counter()` for high-resolution wall-clock timing

The handler performs:
1. **Warmup**: One untimed call to initialize caches and JIT compilation
2. **Auto-calibration**: Doubles runs per block until block time exceeds 5ms
3. **Measurement**: Collects per-run times until minimum run time or run count is reached
4. **Memory tracking**: Records peak GPU memory via `torch.cuda.max_memory_allocated()` (GPU only)

### sigpy GPU

`CupyHandler.blocked_autorange()` provides equivalent functionality for CuPy-based operations:

- Uses `cp.cuda.Event` for GPU timing with `event.synchronize()`
- Uses `cp.get_default_memory_pool().total_bytes()` for peak memory tracking
- Follows the same auto-calibration and measurement algorithm as `TorchHandler`

### Minimum Run Time Estimation

The benchmark suite dynamically estimates `min_run_time` via a pilot run to ensure at least 10 runs complete within a reasonable total time (~5 seconds):

1. **Pilot run**: Measures per-call time with one untimed warmup call followed by one timed call
2. **Estimation**: Sets `min_run_time` based on per-call time:
   - Slow functions (>10ms): `min_run_time = per_call_time * 1.5`
   - Medium functions (>1ms): `min_run_time = per_call_time * 5`
   - Fast functions (<1ms): `min_run_time = 0.1`
3. **Clamping**: Final value is clamped to \[0.01, 0.5\] seconds

If `min_run_time` is explicitly provided, it is used directly without estimation.

### Data Generation in Timing Loop

The handlers' `blocked_autorange()` method accepts an optional `data_gen_fn` parameter. When provided:

- **Data generation is untimed**: `data = data_gen_fn()` runs outside the timing loop
- **Operation is timed**: `fn(data)` runs inside the timing loop
- **Fresh data per iteration**: Prevents GPU L2 cache effects that would artificially speed up repeated computation on the same tensor

This ensures that only the operation itself is measured, while still benefiting from fresh data on each iteration to avoid cache artifacts.

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
