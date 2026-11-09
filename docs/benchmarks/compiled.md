<!---
WARNING: This file is auto-generated.
Do not edit manually — changes will be overwritten on the next build.
-->

# Compiled Benchmarks

Eager vs `torch.compile` performance and compatibility for torch-named-linops operators. Graph-break counts come from `torch._dynamo.explain` and are PyTorch-version dependent.

## Metadata

- **Level**: reduced
- **Date**: 2026-11-05T22:34:44.781289+00:00
- **PyTorch**: 2.15.0+cu132
- **CUDA**: 13.2
- **GPU**: NVIDIA GeForce RTX 3090
- **Python**: 3.13.3
- **Commit**: `1b343e0`
- **Timing protocol**: 5 warmup, 20 measured iterations

## Compatibility Matrix

| Operator | Level | Method | Device | Config | Class | Graph breaks | Max rel diff | Error |
|---|---|---|---|---|---|---|---|---|
| Add | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Add | wrapper | adj | cuda | default | works | 0 | 0 |  |
| Add | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Add | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| ArrayToBlocks | wrapper | adj | cpu | default | works | 0 | 0 |  |
| ArrayToBlocks | wrapper | adj | cuda | default | works | 0 | 0 |  |
| ArrayToBlocks | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| ArrayToBlocks | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| BlocksToArray | wrapper | adj | cpu | default | works | 0 | 0 |  |
| BlocksToArray | wrapper | adj | cuda | default | works | 0 | 0 |  |
| BlocksToArray | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| BlocksToArray | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| Chain | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Chain | wrapper | adj | cuda | default | works | 0 | 0 |  |
| Chain | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Chain | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| Concat | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Concat | wrapper | adj | cuda | default | works_with_breaks | 1 | 0 |  |
| Concat | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Concat | wrapper | fwd | cuda | default | works_with_breaks | 1 | 0 |  |
| Convolution | inner | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Convolution | inner | adj | cuda | default | compiles_slower | 0 | 0 |  |
| Convolution | inner | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Convolution | inner | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| Convolution | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Convolution | wrapper | adj | cuda | default | works | 0 | 0 |  |
| Convolution | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Convolution | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| Dense | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Dense | wrapper | adj | cuda | default | works | 0 | 0 |  |
| Dense | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Dense | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| DenseSimple | wrapper | adj | cpu | default | works | 0 | 0 |  |
| DenseSimple | wrapper | adj | cuda | default | works | 0 | 0 |  |
| DenseSimple | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| DenseSimple | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| Diagonal | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Diagonal | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| Diagonal | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Diagonal | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| FFT | inner | adj | cpu | default | compiles_slower | 0 | 0 |  |
| FFT | inner | adj | cuda | default | compiles_slower | 0 | 0 |  |
| FFT | inner | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| FFT | inner | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| FFT | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| FFT | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| FFT | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| FFT | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| FastSamplingNUFFT | inner | adj | cpu | default | works_with_breaks | 3 | 0 |  |
| FastSamplingNUFFT | inner | adj | cuda | default | works_with_breaks | 3 | 0 |  |
| FastSamplingNUFFT | inner | fwd | cpu | default | works | 0 | 0 |  |
| FastSamplingNUFFT | inner | fwd | cuda | default | works | 0 | 0 |  |
| FastSamplingNUFFT | wrapper | adj | cpu | default | works_with_breaks | 3 | 0 |  |
| FastSamplingNUFFT | wrapper | adj | cuda | default | works_with_breaks | 4 | 0 |  |
| FastSamplingNUFFT | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| FastSamplingNUFFT | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| Identity | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Identity | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| Identity | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Identity | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| Interpolate | inner | adj | cpu | default | works | 0 | 0 |  |
| Interpolate | inner | adj | cuda | default | works | 0 | 1.14e-07 |  |
| Interpolate | inner | fwd | cpu | default | works | 0 | 0 |  |
| Interpolate | inner | fwd | cuda | default | works | 0 | 0 |  |
| Interpolate | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Interpolate | wrapper | adj | cuda | default | works | 0 | 0 |  |
| Interpolate | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Interpolate | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| NUFFT | inner | adj | cpu | default | works | 0 | 0 |  |
| NUFFT | inner | adj | cuda | default | compiles_slower | 0 | 1.01e-06 |  |
| NUFFT | inner | fwd | cpu | default | works | 0 | 0 |  |
| NUFFT | inner | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| NUFFT | wrapper | adj | cpu | default | works | 0 | 0 |  |
| NUFFT | wrapper | adj | cuda | default | works | 0 | 0 |  |
| NUFFT | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| NUFFT | wrapper | fwd | cuda | default | silent_degradation | 0 | 0 |  |
| Pad | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Pad | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| Pad | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Pad | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| PadDim | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| PadDim | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| PadDim | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| PadDim | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| Rearrange | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Rearrange | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| Rearrange | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Rearrange | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| Repeat | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Repeat | wrapper | adj | cuda | default | works | 0 | 0 |  |
| Repeat | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Repeat | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| Sampling | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Sampling | wrapper | adj | cuda | default | works | 0 | 0 |  |
| Sampling | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Sampling | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| SamplingNUFFT | inner | adj | cpu | default | works_with_breaks | 1 | 0 |  |
| SamplingNUFFT | inner | adj | cuda | default | works_with_breaks | 1 | 0 |  |
| SamplingNUFFT | inner | fwd | cpu | default | works | 0 | 0 |  |
| SamplingNUFFT | inner | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| SamplingNUFFT | wrapper | adj | cpu | default | works_with_breaks | 1 | 0 |  |
| SamplingNUFFT | wrapper | adj | cuda | default | works_with_breaks | 2 | 0 |  |
| SamplingNUFFT | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| SamplingNUFFT | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| Scalar | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Scalar | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| Scalar | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Scalar | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| ShapeSpec | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| ShapeSpec | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| ShapeSpec | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| ShapeSpec | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| SimpleLinop | wrapper | adj | cpu | default | works | 0 | 0 |  |
| SimpleLinop | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| SimpleLinop | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| SimpleLinop | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| Sleep | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Sleep | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| Sleep | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Sleep | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| Stack | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Stack | wrapper | adj | cuda | default | works_with_breaks | 1 | 0 |  |
| Stack | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Stack | wrapper | fwd | cuda | default | works_with_breaks | 1 | 0 |  |
| SumReduce | wrapper | adj | cpu | default | works | 0 | 0 |  |
| SumReduce | wrapper | adj | cuda | default | works | 0 | 0 |  |
| SumReduce | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| SumReduce | wrapper | fwd | cuda | default | compiles_slower | 0 | 2.92e-07 |  |
| ToDevice | wrapper | adj | cpu | default | works | 0 | 0 |  |
| ToDevice | wrapper | adj | cuda | default | works | 0 | 0 |  |
| ToDevice | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| ToDevice | wrapper | fwd | cuda | default | works | 0 | 0 |  |
| ToeplitzPSF | inner | adj | cpu | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | inner | adj | cuda | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | inner | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | inner | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| Truncate | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Truncate | wrapper | adj | cuda | default | works | 0 | 0 |  |
| Truncate | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Truncate | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| Zero | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Zero | wrapper | adj | cuda | default | compiles_slower | 0 | 0 |  |
| Zero | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Zero | wrapper | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| functional.fold | inner | fwd | cpu | default | works | 0 | 0 |  |
| functional.fold | inner | fwd | cuda | default | works | 0 | 0 |  |
| functional.grid | inner | fwd | cpu | default | works | 0 | 0 |  |
| functional.grid | inner | fwd | cuda | default | compiles_slower | 0 | 0 |  |
| functional.unfold | inner | fwd | cpu | default | works | 0 | 0 |  |
| functional.unfold | inner | fwd | cuda | default | works | 0 | 0 |  |
| functional.ungrid | inner | fwd | cpu | default | works | 0 | 0 |  |
| functional.ungrid | inner | fwd | cuda | default | compiles_slower | 0 | 0 |  |

### Graph-break details

- **Concat** (wrapper, adj, cuda, default) — 1 breaks
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/namedlinop.py:151 in forward: Unsupported method call`
- **Concat** (wrapper, fwd, cpu, default) — 0 breaks
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/concat.py:173 in _fn: generic_jump TensorVariable()`
- **Concat** (wrapper, fwd, cuda, default) — 1 breaks
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/namedlinop.py:151 in forward: Data-dependent branching`
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/concat.py:173 in _fn: generic_jump TensorVariable()`
- **FastSamplingNUFFT** (wrapper, adj, cpu, default) — 3 breaks
- **FastSamplingNUFFT** (inner, adj, cpu, default) — 3 breaks
- **FastSamplingNUFFT** (wrapper, adj, cuda, default) — 4 breaks
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/namedlinop.py:151 in forward: Data-dependent branching`
- **FastSamplingNUFFT** (inner, adj, cuda, default) — 3 breaks
- **SamplingNUFFT** (wrapper, adj, cpu, default) — 1 breaks
- **SamplingNUFFT** (inner, adj, cpu, default) — 1 breaks
- **SamplingNUFFT** (wrapper, adj, cuda, default) — 2 breaks
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/namedlinop.py:151 in forward: Data-dependent branching`
- **SamplingNUFFT** (inner, adj, cuda, default) — 1 breaks
- **Stack** (wrapper, adj, cpu, default) — 0 breaks
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/stack.py:213 in _fn: Unsupported method call`
- **Stack** (wrapper, adj, cuda, default) — 1 breaks
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/namedlinop.py:151 in forward: Unsupported method call`
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/stack.py:213 in _fn: Unsupported method call`
- **Stack** (wrapper, fwd, cuda, default) — 1 breaks
  - `/home/mark/.local/share/opencode/worktree/d3d230/compiled-benchmarks/src/torchlinops/linops/namedlinop.py:151 in forward: Unsupported method call`

## Speedups

| Operator | Level | Method | Device | Dtype | Eager | Compiled | Speedup | Class |
|---|---|---|---|---|---|---|---|---|
| Sleep | wrapper | fwd | cuda | torch.float32 | 492.68 us | 24.10 us | 20.45x | works |
| ArrayToBlocks | wrapper | fwd | cpu | torch.complex64 | 105.12 us | 15.36 us | 6.84x | works |
| ArrayToBlocks | wrapper | adj | cpu | torch.complex64 | 124.98 us | 20.17 us | 6.20x | works |
| functional.fold | inner | fwd | cpu | torch.complex64 | 96.52 us | 17.79 us | 5.42x | works |
| Add | wrapper | fwd | cpu | torch.float32 | 74.04 us | 13.79 us | 5.37x | works |
| Add | wrapper | adj | cpu | torch.float32 | 72.42 us | 13.58 us | 5.33x | works |
| functional.unfold | inner | fwd | cpu | torch.complex64 | 72.17 us | 13.70 us | 5.27x | works |
| Chain | wrapper | fwd | cpu | torch.float32 | 60.93 us | 12.62 us | 4.83x | works |
| BlocksToArray | wrapper | fwd | cpu | torch.complex64 | 88.63 us | 18.36 us | 4.83x | works |
| BlocksToArray | wrapper | adj | cpu | torch.complex64 | 80.11 us | 16.97 us | 4.72x | works |
| Chain | wrapper | adj | cpu | torch.float32 | 58.81 us | 12.83 us | 4.58x | works |
| Interpolate | wrapper | fwd | cpu | torch.complex64 | 189.24 us | 47.45 us | 3.99x | works |
| Interpolate | inner | fwd | cpu | torch.complex64 | 173.38 us | 45.94 us | 3.77x | works |
| Interpolate | wrapper | adj | cpu | torch.complex64 | 196.90 us | 52.41 us | 3.76x | works |
| Interpolate | inner | adj | cpu | torch.complex64 | 175.63 us | 49.39 us | 3.56x | works |
| Sampling | wrapper | fwd | cuda | torch.float32 | 63.69 us | 22.21 us | 2.87x | works |
| functional.ungrid | inner | fwd | cpu | torch.complex64 | 118.57 us | 43.54 us | 2.72x | works |
| Sampling | wrapper | fwd | cpu | torch.float32 | 28.83 us | 10.76 us | 2.68x | works |
| ArrayToBlocks | wrapper | fwd | cuda | torch.complex64 | 101.40 us | 40.50 us | 2.50x | works |
| functional.grid | inner | fwd | cpu | torch.complex64 | 116.55 us | 47.46 us | 2.46x | works |
| Add | wrapper | fwd | cuda | torch.float32 | 165.84 us | 68.52 us | 2.42x | works |
| ArrayToBlocks | wrapper | adj | cuda | torch.complex64 | 95.49 us | 39.75 us | 2.40x | works |
| Add | wrapper | adj | cuda | torch.float32 | 165.31 us | 70.02 us | 2.36x | works |
| Chain | wrapper | adj | cuda | torch.float32 | 131.51 us | 56.19 us | 2.34x | works |
| Chain | wrapper | fwd | cuda | torch.float32 | 124.93 us | 55.78 us | 2.24x | works |
| BlocksToArray | wrapper | fwd | cuda | torch.complex64 | 90.07 us | 43.42 us | 2.07x | works |
| BlocksToArray | wrapper | adj | cuda | torch.complex64 | 101.44 us | 49.83 us | 2.04x | works |
| FastSamplingNUFFT | wrapper | fwd | cuda | torch.complex64 | 180.47 us | 94.80 us | 1.90x | works |
| Dense | wrapper | fwd | cpu | torch.float32 | 18.72 us | 10.77 us | 1.74x | works |
| DenseSimple | wrapper | adj | cpu | torch.float32 | 18.26 us | 10.52 us | 1.74x | works |
| DenseSimple | wrapper | fwd | cpu | torch.float32 | 17.96 us | 10.53 us | 1.71x | works |
| Interpolate | wrapper | adj | cuda | torch.complex64 | 87.01 us | 52.92 us | 1.64x | works |
| FastSamplingNUFFT | inner | fwd | cuda | torch.complex64 | 135.95 us | 83.16 us | 1.63x | works |
| Dense | wrapper | adj | cpu | torch.float32 | 17.85 us | 11.04 us | 1.62x | works |
| Interpolate | wrapper | fwd | cuda | torch.complex64 | 81.22 us | 51.18 us | 1.59x | works |
| Interpolate | inner | fwd | cuda | torch.complex64 | 72.44 us | 46.76 us | 1.55x | works |
| Interpolate | inner | adj | cuda | torch.complex64 | 71.75 us | 47.05 us | 1.52x | works |
| functional.unfold | inner | fwd | cuda | torch.complex64 | 56.40 us | 38.40 us | 1.47x | works |
| DenseSimple | wrapper | fwd | cuda | torch.float32 | 40.55 us | 29.28 us | 1.38x | works |
| DenseSimple | wrapper | adj | cuda | torch.float32 | 40.85 us | 29.52 us | 1.38x | works |
| Dense | wrapper | adj | cuda | torch.float32 | 40.95 us | 29.61 us | 1.38x | works |
| FastSamplingNUFFT | wrapper | fwd | cpu | torch.complex64 | 291.41 us | 214.53 us | 1.36x | works |
| NUFFT | wrapper | adj | cpu | torch.complex64 | 672.83 us | 504.05 us | 1.33x | works |
| NUFFT | wrapper | fwd | cpu | torch.complex64 | 666.79 us | 504.43 us | 1.32x | works |
| NUFFT | inner | fwd | cpu | torch.complex64 | 654.74 us | 503.87 us | 1.30x | works |
| Dense | wrapper | fwd | cuda | torch.float32 | 37.79 us | 29.18 us | 1.30x | works |
| Sampling | wrapper | adj | cpu | torch.float32 | 42.29 us | 33.04 us | 1.28x | works |
| functional.fold | inner | fwd | cuda | torch.complex64 | 49.09 us | 38.37 us | 1.28x | works |
| SamplingNUFFT | wrapper | fwd | cuda | torch.complex64 | 469.02 us | 372.88 us | 1.26x | works |
| NUFFT | inner | adj | cpu | torch.complex64 | 632.88 us | 508.25 us | 1.25x | works |
| FastSamplingNUFFT | inner | fwd | cpu | torch.complex64 | 265.86 us | 219.09 us | 1.21x | works |
| Sampling | wrapper | adj | cuda | torch.float32 | 149.38 us | 128.13 us | 1.17x | works |
| SumReduce | wrapper | fwd | cpu | torch.float32 | 12.11 us | 10.42 us | 1.16x | works |
| Repeat | wrapper | adj | cpu | torch.float32 | 12.21 us | 10.51 us | 1.16x | works |
| SimpleLinop | wrapper | fwd | cpu | torch.float32 | 9.41 us | 8.22 us | 1.15x | works |
| SimpleLinop | wrapper | adj | cpu | torch.float32 | 9.59 us | 8.41 us | 1.14x | works |
| ToDevice | wrapper | fwd | cpu | torch.float32 | 8.05 us | 7.22 us | 1.12x | works |
| Pad | wrapper | fwd | cpu | torch.float32 | 10.27 us | 9.37 us | 1.10x | works |
| Truncate | wrapper | adj | cpu | torch.float32 | 10.18 us | 9.40 us | 1.08x | works |
| ToDevice | wrapper | adj | cpu | torch.float32 | 8.02 us | 7.41 us | 1.08x | works |
| Convolution | wrapper | adj | cpu | torch.float32 | 50.79 us | 48.20 us | 1.05x | works |
| Convolution | wrapper | fwd | cpu | torch.float32 | 43.76 us | 42.06 us | 1.04x | works |
| Stack | wrapper | adj | cuda | torch.float32 | 856.54 us | 825.40 us | 1.04x | works_with_breaks |
| SumReduce | wrapper | adj | cpu | torch.float32 | 14.37 us | 13.94 us | 1.03x | works |
| Convolution | wrapper | adj | cuda | torch.float32 | 30.28 us | 30.33 us | 1.00x | works |
| Repeat | wrapper | fwd | cpu | torch.float32 | 13.73 us | 13.82 us | 0.99x | works |
| SamplingNUFFT | wrapper | fwd | cpu | torch.complex64 | 426.31 us | 439.20 us | 0.97x | works |
| Concat | wrapper | fwd | cuda | torch.float32 | 747.33 us | 771.54 us | 0.97x | works_with_breaks |
| Concat | wrapper | adj | cuda | torch.float32 | 717.40 us | 742.31 us | 0.97x | works_with_breaks |
| Repeat | wrapper | adj | cuda | torch.float32 | 21.92 us | 22.79 us | 0.96x | works |
| Stack | wrapper | adj | cpu | torch.float32 | 456.79 us | 477.86 us | 0.96x | works |
| PadDim | wrapper | fwd | cpu | torch.float32 | 8.67 us | 9.08 us | 0.96x | works |
| Sleep | wrapper | fwd | cpu | torch.float32 | 57.87 us | 60.85 us | 0.95x | works |
| NUFFT | wrapper | fwd | cuda | torch.complex64 | 380.97 us | 402.58 us | 0.95x | silent_degradation |
| Concat | wrapper | adj | cpu | torch.float32 | 413.45 us | 442.36 us | 0.93x | works |
| SumReduce | wrapper | adj | cuda | torch.float32 | 20.66 us | 22.14 us | 0.93x | works |
| ToDevice | wrapper | fwd | cuda | torch.float32 | 11.91 us | 12.87 us | 0.93x | works |
| Concat | wrapper | fwd | cpu | torch.float32 | 431.33 us | 468.39 us | 0.92x | works |
| PadDim | wrapper | fwd | cuda | torch.float32 | 17.28 us | 18.79 us | 0.92x | works |
| Truncate | wrapper | adj | cuda | torch.float32 | 17.98 us | 19.59 us | 0.92x | works |
| Stack | wrapper | fwd | cuda | torch.float32 | 691.38 us | 753.86 us | 0.92x | works_with_breaks |
| Stack | wrapper | fwd | cpu | torch.float32 | 421.67 us | 460.93 us | 0.91x | works |
| ToDevice | wrapper | adj | cuda | torch.float32 | 12.12 us | 13.25 us | 0.91x | works |
| SamplingNUFFT | inner | fwd | cpu | torch.complex64 | 401.99 us | 440.63 us | 0.91x | works |
| NUFFT | wrapper | adj | cuda | torch.complex64 | 387.74 us | 428.93 us | 0.90x | works |
| Pad | wrapper | fwd | cuda | torch.float32 | 18.54 us | 20.58 us | 0.90x | works |
| Convolution | inner | adj | cpu | torch.float32 | 38.36 us | 42.83 us | 0.90x | compiles_slower |
| Repeat | wrapper | fwd | cuda | torch.float32 | 19.09 us | 21.64 us | 0.88x | compiles_slower |
| SimpleLinop | wrapper | adj | cuda | torch.float32 | 18.01 us | 20.49 us | 0.88x | compiles_slower |
| SimpleLinop | wrapper | fwd | cuda | torch.float32 | 17.21 us | 19.92 us | 0.86x | compiles_slower |
| NUFFT | inner | fwd | cuda | torch.complex64 | 339.70 us | 394.57 us | 0.86x | compiles_slower |
| Convolution | wrapper | fwd | cuda | torch.float32 | 25.91 us | 30.43 us | 0.85x | compiles_slower |
| Scalar | wrapper | fwd | cpu | torch.float32 | 7.26 us | 8.55 us | 0.85x | compiles_slower |
| Scalar | wrapper | adj | cpu | torch.float32 | 7.42 us | 8.75 us | 0.85x | compiles_slower |
| SumReduce | wrapper | fwd | cuda | torch.float32 | 20.19 us | 23.84 us | 0.85x | compiles_slower |
| Convolution | inner | fwd | cpu | torch.float32 | 33.56 us | 40.25 us | 0.83x | compiles_slower |
| Zero | wrapper | adj | cpu | torch.float32 | 6.97 us | 8.51 us | 0.82x | compiles_slower |
| Zero | wrapper | fwd | cpu | torch.float32 | 6.89 us | 8.47 us | 0.81x | compiles_slower |
| SamplingNUFFT | wrapper | adj | cpu | torch.complex64 | 442.04 us | 544.31 us | 0.81x | works_with_breaks |
| ToeplitzPSF | wrapper | adj | cpu | torch.complex64 | 20.35 us | 25.09 us | 0.81x | compiles_slower |
| functional.grid | inner | fwd | cuda | torch.complex64 | 39.24 us | 48.43 us | 0.81x | compiles_slower |
| functional.ungrid | inner | fwd | cuda | torch.complex64 | 37.96 us | 46.92 us | 0.81x | compiles_slower |
| NUFFT | inner | adj | cuda | torch.complex64 | 318.51 us | 396.26 us | 0.80x | compiles_slower |
| Diagonal | wrapper | adj | cpu | torch.float32 | 7.62 us | 9.54 us | 0.80x | compiles_slower |
| Diagonal | wrapper | fwd | cpu | torch.float32 | 7.27 us | 9.16 us | 0.79x | compiles_slower |
| FastSamplingNUFFT | wrapper | adj | cpu | torch.complex64 | 329.06 us | 415.92 us | 0.79x | works_with_breaks |
| Rearrange | wrapper | adj | cpu | torch.float32 | 10.05 us | 12.82 us | 0.78x | compiles_slower |
| Diagonal | wrapper | fwd | cuda | torch.float32 | 14.12 us | 18.34 us | 0.77x | compiles_slower |
| ToeplitzPSF | wrapper | fwd | cpu | torch.complex64 | 17.39 us | 22.74 us | 0.76x | compiles_slower |
| SamplingNUFFT | inner | fwd | cuda | torch.complex64 | 271.78 us | 356.83 us | 0.76x | compiles_slower |
| FastSamplingNUFFT | inner | adj | cpu | torch.complex64 | 278.96 us | 366.29 us | 0.76x | works_with_breaks |
| SamplingNUFFT | inner | adj | cpu | torch.complex64 | 400.15 us | 526.20 us | 0.76x | works_with_breaks |
| Rearrange | wrapper | fwd | cpu | torch.float32 | 9.51 us | 12.52 us | 0.76x | compiles_slower |
| Identity | wrapper | adj | cuda | torch.float32 | 9.69 us | 12.89 us | 0.75x | compiles_slower |
| Scalar | wrapper | fwd | cuda | torch.float32 | 13.86 us | 18.51 us | 0.75x | compiles_slower |
| Diagonal | wrapper | adj | cuda | torch.float32 | 14.17 us | 18.94 us | 0.75x | compiles_slower |
| Sleep | wrapper | adj | cuda | torch.float32 | 9.29 us | 12.58 us | 0.74x | compiles_slower |
| Pad | wrapper | adj | cpu | torch.float32 | 8.45 us | 11.52 us | 0.73x | compiles_slower |
| ToeplitzPSF | wrapper | fwd | cuda | torch.complex64 | 27.63 us | 37.87 us | 0.73x | compiles_slower |
| ShapeSpec | wrapper | adj | cuda | torch.float32 | 9.93 us | 13.87 us | 0.72x | compiles_slower |
| Zero | wrapper | fwd | cuda | torch.float32 | 12.36 us | 17.29 us | 0.71x | compiles_slower |
| Truncate | wrapper | fwd | cpu | torch.float32 | 7.80 us | 10.97 us | 0.71x | compiles_slower |
| ToeplitzPSF | wrapper | adj | cuda | torch.complex64 | 31.36 us | 44.47 us | 0.71x | compiles_slower |
| Identity | wrapper | fwd | cuda | torch.float32 | 9.14 us | 13.04 us | 0.70x | compiles_slower |
| ShapeSpec | wrapper | fwd | cpu | torch.float32 | 6.38 us | 9.18 us | 0.70x | compiles_slower |
| Scalar | wrapper | adj | cuda | torch.float32 | 13.59 us | 19.66 us | 0.69x | compiles_slower |
| PadDim | wrapper | adj | cpu | torch.float32 | 7.58 us | 10.98 us | 0.69x | compiles_slower |
| Rearrange | wrapper | fwd | cuda | torch.float32 | 13.84 us | 20.19 us | 0.69x | compiles_slower |
| Convolution | inner | adj | cuda | torch.float32 | 17.94 us | 26.18 us | 0.69x | compiles_slower |
| FastSamplingNUFFT | wrapper | adj | cuda | torch.complex64 | 273.02 us | 402.65 us | 0.68x | works_with_breaks |
| ShapeSpec | wrapper | fwd | cuda | torch.float32 | 8.83 us | 13.04 us | 0.68x | compiles_slower |
| Rearrange | wrapper | adj | cuda | torch.float32 | 14.20 us | 21.05 us | 0.67x | compiles_slower |
| Identity | wrapper | fwd | cpu | torch.float32 | 6.21 us | 9.23 us | 0.67x | compiles_slower |
| Sleep | wrapper | adj | cpu | torch.float32 | 5.95 us | 8.88 us | 0.67x | compiles_slower |
| ShapeSpec | wrapper | adj | cpu | torch.float32 | 6.58 us | 9.84 us | 0.67x | compiles_slower |
| FastSamplingNUFFT | inner | adj | cuda | torch.complex64 | 219.79 us | 334.14 us | 0.66x | works_with_breaks |
| SamplingNUFFT | wrapper | adj | cuda | torch.complex64 | 400.55 us | 609.92 us | 0.66x | works_with_breaks |
| Identity | wrapper | adj | cpu | torch.float32 | 6.31 us | 9.66 us | 0.65x | compiles_slower |
| Zero | wrapper | adj | cuda | torch.float32 | 11.75 us | 18.21 us | 0.65x | compiles_slower |
| Pad | wrapper | adj | cuda | torch.float32 | 12.29 us | 19.28 us | 0.64x | compiles_slower |
| Convolution | inner | fwd | cuda | torch.float32 | 16.19 us | 25.97 us | 0.62x | compiles_slower |
| PadDim | wrapper | adj | cuda | torch.float32 | 10.85 us | 17.73 us | 0.61x | compiles_slower |
| SamplingNUFFT | inner | adj | cuda | torch.complex64 | 331.52 us | 551.95 us | 0.60x | works_with_breaks |
| Truncate | wrapper | fwd | cuda | torch.float32 | 10.99 us | 18.33 us | 0.60x | compiles_slower |
| ToeplitzPSF | inner | fwd | cpu | torch.complex64 | 10.21 us | 19.93 us | 0.51x | compiles_slower |
| ToeplitzPSF | inner | adj | cpu | torch.complex64 | 13.66 us | 27.49 us | 0.50x | compiles_slower |
| ToeplitzPSF | inner | adj | cuda | torch.complex64 | 22.70 us | 46.43 us | 0.49x | compiles_slower |
| ToeplitzPSF | inner | fwd | cuda | torch.complex64 | 16.62 us | 35.15 us | 0.47x | compiles_slower |
| FFT | wrapper | adj | cpu | torch.complex64 | 86.15 us | 193.64 us | 0.44x | compiles_slower |
| FFT | wrapper | fwd | cpu | torch.complex64 | 83.77 us | 190.71 us | 0.44x | compiles_slower |
| FFT | inner | adj | cpu | torch.complex64 | 78.78 us | 184.53 us | 0.43x | compiles_slower |
| FFT | wrapper | fwd | cuda | torch.complex64 | 131.11 us | 309.87 us | 0.42x | compiles_slower |
| FFT | wrapper | adj | cuda | torch.complex64 | 131.31 us | 311.65 us | 0.42x | compiles_slower |
| FFT | inner | fwd | cpu | torch.complex64 | 74.79 us | 184.78 us | 0.40x | compiles_slower |
| FFT | inner | fwd | cuda | torch.complex64 | 121.75 us | 303.78 us | 0.40x | compiles_slower |
| FFT | inner | adj | cuda | torch.complex64 | 121.45 us | 305.43 us | 0.40x | compiles_slower |

## Charts

### CPU

![Compiled speedup — CPU](assets/speedup_cpu.png)

![Wrapper vs inner compile unit — CPU](assets/speedup_wrapper_vs_inner_cpu.png)

### GPU

![Compiled speedup — GPU](assets/speedup_cuda.png)

![Wrapper vs inner compile unit — GPU](assets/speedup_wrapper_vs_inner_cuda.png)

## Summary

### Effective classes

| Class | Rows |
|---|---|
| compiles_slower | 62 |
| silent_degradation | 1 |
| works | 81 |
| works_with_breaks | 12 |

### Rows by device

| Device | Rows |
|---|---|
| cpu | 78 |
| cuda | 78 |

## Functional Kernels

| Operator | Device | Dtype | Config | Class | Graph breaks | Speedup |
|---|---|---|---|---|---|---|
| functional.fold | cpu | torch.complex64 | default | works | 0 | 5.42x |
| functional.fold | cuda | torch.complex64 | default | works | 0 | 1.28x |
| functional.grid | cpu | torch.complex64 | default | works | 0 | 2.46x |
| functional.grid | cuda | torch.complex64 | default | compiles_slower | 0 | 0.81x |
| functional.unfold | cpu | torch.complex64 | default | works | 0 | 5.27x |
| functional.unfold | cuda | torch.complex64 | default | works | 0 | 1.47x |
| functional.ungrid | cpu | torch.complex64 | default | works | 0 | 2.72x |
| functional.ungrid | cuda | torch.complex64 | default | compiles_slower | 0 | 0.81x |

## Recompilation

These sections require the full suite.

## Autograd

These sections require the full suite.

## Pipeline

These sections require the full suite.

## Caveats

- Timings are steady-state per-call medians and exclude one-time compilation cost.
- Problem shapes are small; speedup ratios are shape-dependent and should not be extrapolated to production sizes.
- Results come from one GPU and one PyTorch version. Re-run after upgrading PyTorch.
- Dynamic-shape behavior is sampled only by the full suite's recompile phase.
