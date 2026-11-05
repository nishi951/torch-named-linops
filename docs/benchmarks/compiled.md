<!---
WARNING: This file is auto-generated.
Do not edit manually — changes will be overwritten on the next build.
-->

# Compiled Benchmarks

Eager vs `torch.compile` performance and compatibility for torch-named-linops operators. Graph-break counts come from `torch._dynamo.explain` and are PyTorch-version dependent.

## Metadata

- **Level**: reduced
- **Date**: 2026-11-05T19:40:14.286333+00:00
- **PyTorch**: 2.15.0+cu132
- **CUDA**: 13.2
- **GPU**: NVIDIA GeForce RTX 3090
- **Python**: 3.13.3
- **Commit**: `ff25320`
- **Timing protocol**: 5 warmup, 20 measured iterations

## Compatibility Matrix

| Operator | Level | Method | Device | Config | Class | Graph breaks | Max rel diff | Error |
|---|---|---|---|---|---|---|---|---|
| Add | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Add | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| ArrayToBlocks | wrapper | adj | cpu | default | fails_loud | — | — | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| ArrayToBlocks | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| BlocksToArray | wrapper | adj | cpu | default | fails_loud | — | — | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| BlocksToArray | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Chain | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Chain | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Concat | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Concat | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Convolution | inner | adj | cpu | default | works | 0 | 0 |  |
| Convolution | inner | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Convolution | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Convolution | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Dense | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Dense | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| DenseSimple | wrapper | adj | cpu | default | works | 0 | 0 |  |
| DenseSimple | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Diagonal | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Diagonal | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| FFT | inner | adj | cpu | default | compiles_slower | 0 | 0 |  |
| FFT | inner | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| FFT | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| FFT | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| FastSamplingNUFFT | inner | adj | cpu | default | works_with_breaks | 3 | 0 |  |
| FastSamplingNUFFT | inner | fwd | cpu | default | works | 0 | 0 |  |
| FastSamplingNUFFT | wrapper | adj | cpu | default | works_with_breaks | 3 | 0 |  |
| FastSamplingNUFFT | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Identity | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Identity | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Interpolate | inner | adj | cpu | default | works | 0 | 0 |  |
| Interpolate | inner | fwd | cpu | default | works | 0 | 0 |  |
| Interpolate | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Interpolate | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| NUFFT | inner | adj | cpu | default | works | 0 | 0 |  |
| NUFFT | inner | fwd | cpu | default | works | 0 | 0 |  |
| NUFFT | wrapper | adj | cpu | default | works | 0 | 0 |  |
| NUFFT | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Pad | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Pad | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| PadDim | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| PadDim | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Rearrange | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Rearrange | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Repeat | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Repeat | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Sampling | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Sampling | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| SamplingNUFFT | inner | adj | cpu | default | works_with_breaks | 1 | 0 |  |
| SamplingNUFFT | inner | fwd | cpu | default | works | 0 | 0 |  |
| SamplingNUFFT | wrapper | adj | cpu | default | works_with_breaks | 1 | 0 |  |
| SamplingNUFFT | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Scalar | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Scalar | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| ShapeSpec | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| ShapeSpec | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| SimpleLinop | wrapper | adj | cpu | default | works | 0 | 0 |  |
| SimpleLinop | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Sleep | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Sleep | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| Stack | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Stack | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| SumReduce | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| SumReduce | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| ToDevice | wrapper | adj | cpu | default | works | 0 | 0 |  |
| ToDevice | wrapper | fwd | cpu | default | works | 0 | 0 |  |
| ToeplitzPSF | inner | adj | cpu | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | inner | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| ToeplitzPSF | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Truncate | wrapper | adj | cpu | default | works | 0 | 0 |  |
| Truncate | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| Zero | wrapper | adj | cpu | default | compiles_slower | 0 | 0 |  |
| Zero | wrapper | fwd | cpu | default | compiles_slower | 0 | 0 |  |
| functional.fold | inner | fwd | cpu | default | works | 0 | 0 |  |
| functional.grid | inner | fwd | cpu | default | works | 0 | 0 |  |
| functional.unfold | inner | fwd | cpu | default | works | 0 | 0 |  |
| functional.ungrid | inner | fwd | cpu | default | works | 0 | 0 |  |

## Speedups

| Operator | Level | Method | Device | Eager | Compiled | Speedup | Class |
|---|---|---|---|---|---|---|---|
| ArrayToBlocks | wrapper | fwd | cpu | 109.21 us | 17.31 us | 6.31x | works |
| Add | wrapper | fwd | cpu | 80.05 us | 15.18 us | 5.27x | works |
| Add | wrapper | adj | cpu | 80.06 us | 15.30 us | 5.23x | works |
| functional.unfold | inner | fwd | cpu | 72.31 us | 13.96 us | 5.18x | works |
| functional.fold | inner | fwd | cpu | 92.27 us | 18.64 us | 4.95x | works |
| BlocksToArray | wrapper | fwd | cpu | 97.65 us | 21.31 us | 4.58x | works |
| Chain | wrapper | fwd | cpu | 63.54 us | 14.37 us | 4.42x | works |
| Chain | wrapper | adj | cpu | 61.41 us | 14.80 us | 4.15x | works |
| Interpolate | wrapper | fwd | cpu | 199.58 us | 51.61 us | 3.87x | works |
| Interpolate | inner | fwd | cpu | 193.18 us | 50.85 us | 3.80x | works |
| Interpolate | wrapper | adj | cpu | 206.56 us | 58.95 us | 3.50x | works |
| Interpolate | inner | adj | cpu | 185.36 us | 56.34 us | 3.29x | works |
| Sampling | wrapper | fwd | cpu | 30.97 us | 11.59 us | 2.67x | works |
| functional.ungrid | inner | fwd | cpu | 122.74 us | 47.08 us | 2.61x | works |
| functional.grid | inner | fwd | cpu | 124.53 us | 54.18 us | 2.30x | works |
| DenseSimple | wrapper | fwd | cpu | 18.69 us | 10.53 us | 1.77x | works |
| Dense | wrapper | adj | cpu | 19.04 us | 11.22 us | 1.70x | works |
| DenseSimple | wrapper | adj | cpu | 18.00 us | 11.30 us | 1.59x | works |
| Convolution | inner | adj | cpu | 105.72 us | 69.79 us | 1.51x | works |
| Dense | wrapper | fwd | cpu | 16.70 us | 11.68 us | 1.43x | works |
| NUFFT | wrapper | fwd | cpu | 693.87 us | 522.21 us | 1.33x | works |
| NUFFT | inner | fwd | cpu | 683.35 us | 517.98 us | 1.32x | works |
| FastSamplingNUFFT | wrapper | fwd | cpu | 295.07 us | 229.07 us | 1.29x | works |
| NUFFT | inner | adj | cpu | 662.60 us | 525.85 us | 1.26x | works |
| FastSamplingNUFFT | inner | fwd | cpu | 284.86 us | 226.42 us | 1.26x | works |
| Sampling | wrapper | adj | cpu | 45.76 us | 37.48 us | 1.22x | works |
| SumReduce | wrapper | fwd | cpu | 13.31 us | 11.15 us | 1.19x | works |
| Repeat | wrapper | adj | cpu | 13.49 us | 11.35 us | 1.19x | works |
| SimpleLinop | wrapper | fwd | cpu | 9.58 us | 8.41 us | 1.14x | works |
| ToDevice | wrapper | fwd | cpu | 8.26 us | 7.32 us | 1.13x | works |
| Truncate | wrapper | adj | cpu | 10.85 us | 9.70 us | 1.12x | works |
| Pad | wrapper | fwd | cpu | 10.80 us | 9.76 us | 1.11x | works |
| SimpleLinop | wrapper | adj | cpu | 9.78 us | 8.85 us | 1.10x | works |
| NUFFT | wrapper | adj | cpu | 701.82 us | 644.74 us | 1.09x | works |
| ToDevice | wrapper | adj | cpu | 8.37 us | 7.89 us | 1.06x | works |
| Convolution | wrapper | adj | cpu | 54.97 us | 55.20 us | 1.00x | works |
| Stack | wrapper | fwd | cpu | 439.72 us | 444.37 us | 0.99x | works |
| SamplingNUFFT | wrapper | fwd | cpu | 443.14 us | 459.34 us | 0.96x | works |
| Concat | wrapper | fwd | cpu | 444.06 us | 467.19 us | 0.95x | works |
| Sleep | wrapper | fwd | cpu | 58.19 us | 61.45 us | 0.95x | works |
| Repeat | wrapper | fwd | cpu | 14.99 us | 15.85 us | 0.95x | works |
| Concat | wrapper | adj | cpu | 424.89 us | 449.46 us | 0.95x | works |
| Stack | wrapper | adj | cpu | 445.85 us | 472.74 us | 0.94x | works |
| PadDim | wrapper | fwd | cpu | 9.09 us | 9.69 us | 0.94x | works |
| Convolution | wrapper | fwd | cpu | 45.57 us | 48.56 us | 0.94x | works |
| SamplingNUFFT | inner | fwd | cpu | 422.67 us | 452.03 us | 0.94x | works |
| Zero | wrapper | fwd | cpu | 7.76 us | 8.63 us | 0.90x | compiles_slower |
| SumReduce | wrapper | adj | cpu | 14.76 us | 16.71 us | 0.88x | compiles_slower |
| Scalar | wrapper | fwd | cpu | 7.46 us | 8.83 us | 0.84x | compiles_slower |
| Zero | wrapper | adj | cpu | 7.03 us | 8.41 us | 0.84x | compiles_slower |
| Diagonal | wrapper | adj | cpu | 7.40 us | 9.12 us | 0.81x | compiles_slower |
| SamplingNUFFT | wrapper | adj | cpu | 456.04 us | 566.83 us | 0.80x | works_with_breaks |
| Scalar | wrapper | adj | cpu | 7.35 us | 9.48 us | 0.78x | compiles_slower |
| Diagonal | wrapper | fwd | cpu | 7.45 us | 9.81 us | 0.76x | compiles_slower |
| SamplingNUFFT | inner | adj | cpu | 418.48 us | 552.55 us | 0.76x | works_with_breaks |
| FastSamplingNUFFT | wrapper | adj | cpu | 330.06 us | 447.27 us | 0.74x | works_with_breaks |
| FastSamplingNUFFT | inner | adj | cpu | 284.63 us | 390.72 us | 0.73x | works_with_breaks |
| ToeplitzPSF | wrapper | fwd | cpu | 18.52 us | 25.92 us | 0.71x | compiles_slower |
| Rearrange | wrapper | adj | cpu | 10.38 us | 14.58 us | 0.71x | compiles_slower |
| Rearrange | wrapper | fwd | cpu | 10.36 us | 14.68 us | 0.71x | compiles_slower |
| ShapeSpec | wrapper | adj | cpu | 6.76 us | 9.62 us | 0.70x | compiles_slower |
| Convolution | inner | fwd | cpu | 31.47 us | 46.14 us | 0.68x | compiles_slower |
| ToeplitzPSF | wrapper | adj | cpu | 19.35 us | 28.54 us | 0.68x | compiles_slower |
| Identity | wrapper | adj | cpu | 6.70 us | 9.94 us | 0.67x | compiles_slower |
| Sleep | wrapper | adj | cpu | 6.16 us | 9.16 us | 0.67x | compiles_slower |
| Identity | wrapper | fwd | cpu | 6.10 us | 9.32 us | 0.65x | compiles_slower |
| Pad | wrapper | adj | cpu | 8.76 us | 13.61 us | 0.64x | compiles_slower |
| Truncate | wrapper | fwd | cpu | 7.99 us | 12.57 us | 0.64x | compiles_slower |
| ShapeSpec | wrapper | fwd | cpu | 5.58 us | 9.26 us | 0.60x | compiles_slower |
| PadDim | wrapper | adj | cpu | 7.84 us | 13.10 us | 0.60x | compiles_slower |
| FFT | wrapper | fwd | cpu | 88.56 us | 194.78 us | 0.45x | compiles_slower |
| ToeplitzPSF | inner | fwd | cpu | 10.13 us | 22.72 us | 0.45x | compiles_slower |
| FFT | wrapper | adj | cpu | 88.29 us | 201.62 us | 0.44x | compiles_slower |
| ToeplitzPSF | inner | adj | cpu | 13.40 us | 31.66 us | 0.42x | compiles_slower |
| FFT | inner | fwd | cpu | 80.05 us | 190.11 us | 0.42x | compiles_slower |
| FFT | inner | adj | cpu | 79.34 us | 194.69 us | 0.41x | compiles_slower |

## Summary

### Effective classes

| Class | Rows |
|---|---|
| compiles_slower | 26 |
| fails_loud | 2 |
| works | 46 |
| works_with_breaks | 4 |

### Rows by device

| Device | Rows |
|---|---|
| cpu | 78 |

### Failure signatures

| Operator | Method | Device | Config | Signature |
|---|---|---|---|---|
| ArrayToBlocks | adj | cpu | default | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| BlocksToArray | adj | cpu | default | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |

## Functional Kernels

| Operator | Device | Class | Graph breaks | Speedup |
|---|---|---|---|---|
| functional.fold | cpu | works | 0 | 4.95x |
| functional.grid | cpu | works | 0 | 2.30x |
| functional.unfold | cpu | works | 0 | 5.18x |
| functional.ungrid | cpu | works | 0 | 2.61x |

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
