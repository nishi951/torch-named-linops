# torch.compile compatibility investigation — machine report

Generated from `results/results.json` by `report.py`.
Environment: PyTorch 2.15.0+cu132 (bundled triton), NVIDIA RTX 3090, CUDA 13.2.

## Overview

- Sweep cells: 720 (408 smoke, 312 deep-dive)
- Recompile checks: 41
- Autograd checks: 18
- Pipeline rows: 8

### Effective class counts

| class | cells |
|---|---|
| compiles_slower | 223 |
| fails_loud | 103 |
| works | 365 |
| works_with_breaks | 29 |

### Silent numeric mismatches

None: every compiled cell that ran agreed with eager within 1e-3 rel.

## Wrapper vs inner level

| op | level | cells | works | with_breaks | slower | fails | median_speedup | max_breaks |
|---|---|---|---|---|---|---|---|---|
| Add | wrapper | 18 | 8 | 3 | 4 | 3 | 0.78 | 4 |
| ArrayToBlocks | wrapper | 18 | 10 | 0 | 0 | 8 | 5 | 0 |
| BlocksToArray | wrapper | 18 | 10 | 0 | 0 | 8 | 4.3 | 0 |
| Chain | wrapper | 18 | 8 | 3 | 4 | 3 | 0.88 | 1 |
| Concat | wrapper | 18 | 6 | 0 | 0 | 12 | - | 0 |
| Convolution | inner | 36 | 16 | 0 | 20 | 0 | 0.8 | 0 |
| Convolution | wrapper | 36 | 24 | 0 | 12 | 0 | 0.89 | 0 |
| Dense | wrapper | 18 | 18 | 0 | 0 | 0 | 1.4 | 0 |
| DenseSimple | wrapper | 18 | 18 | 0 | 0 | 0 | 1.4 | 0 |
| Diagonal | wrapper | 18 | 6 | 0 | 12 | 0 | 0.72 | 0 |
| FFT | inner | 18 | 6 | 0 | 12 | 0 | 0.38 | 0 |
| FFT | wrapper | 18 | 6 | 0 | 12 | 0 | 0.44 | 0 |
| FastSamplingNUFFT | inner | 18 | 10 | 4 | 0 | 4 | 1.1 | 4 |
| FastSamplingNUFFT | wrapper | 18 | 8 | 5 | 0 | 5 | 0.81 | 7 |
| Identity | wrapper | 18 | 6 | 0 | 12 | 0 | 0.72 | 0 |
| Interpolate | inner | 18 | 12 | 0 | 0 | 6 | 3.7 | 0 |
| Interpolate | wrapper | 18 | 12 | 0 | 0 | 6 | 3.8 | 0 |
| NUFFT | inner | 18 | 12 | 0 | 0 | 6 | 1.3 | 0 |
| NUFFT | wrapper | 18 | 12 | 0 | 0 | 6 | 1.4 | 0 |
| Pad | wrapper | 18 | 8 | 0 | 10 | 0 | 0.65 | 0 |
| PadDim | wrapper | 18 | 6 | 0 | 12 | 0 | 0.56 | 0 |
| Rearrange | wrapper | 18 | 6 | 0 | 12 | 0 | 0.64 | 0 |
| Repeat | wrapper | 18 | 9 | 1 | 7 | 1 | 0.76 | 1 |
| Sampling | wrapper | 18 | 12 | 2 | 0 | 4 | 2 | 1 |
| SamplingNUFFT | inner | 18 | 7 | 4 | 3 | 4 | 0.74 | 2 |
| SamplingNUFFT | wrapper | 18 | 8 | 5 | 0 | 5 | 0.74 | 6 |
| Scalar | wrapper | 18 | 6 | 0 | 12 | 0 | 0.71 | 0 |
| ShapeSpec | wrapper | 18 | 6 | 0 | 12 | 0 | 0.7 | 0 |
| SimpleLinop | wrapper | 18 | 13 | 0 | 5 | 0 | 0.95 | 0 |
| Sleep | wrapper | 12 | 6 | 0 | 4 | 2 | 0.74 | 0 |
| Stack | wrapper | 18 | 7 | 0 | 1 | 10 | 0.9 | 0 |
| SumReduce | wrapper | 18 | 8 | 1 | 8 | 1 | 0.79 | 1 |
| ToDevice | wrapper | 18 | 12 | 0 | 6 | 0 | 0.9 | 0 |
| ToeplitzPSF | inner | 18 | 6 | 0 | 12 | 0 | 0.52 | 0 |
| ToeplitzPSF | wrapper | 18 | 7 | 0 | 11 | 0 | 0.71 | 0 |
| Truncate | wrapper | 18 | 8 | 1 | 8 | 1 | 0.54 | 1 |
| Zero | wrapper | 18 | 6 | 0 | 12 | 0 | 0.5 | 0 |
| functional.fold | inner | 6 | 4 | 0 | 0 | 2 | 4.9 | 0 |
| functional.grid | inner | 6 | 4 | 0 | 0 | 2 | 2.3 | 0 |
| functional.unfold | inner | 6 | 4 | 0 | 0 | 2 | 5 | 0 |
| functional.ungrid | inner | 6 | 4 | 0 | 0 | 2 | 2.4 | 0 |

## Full cell matrix

| phase | op | method | device | dtype | level | config | class | breaks | rel_diff | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke | Add | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.78 |
| smoke | Add | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Add | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.71 |
| smoke | Add | adj | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.79 |
| smoke | Add | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Add | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Add | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.78 |
| smoke | Add | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Add | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.75 |
| smoke | Add | fwd | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.81 |
| smoke | Add | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Add | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Add | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 1.1 |
| smoke | Add | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Add | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 1.1 |
| smoke | Add | normal | cuda | torch.complex64 | wrapper | default | works_with_breaks | 4 | 0 | 0.73 |
| smoke | Add | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Add | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | ArrayToBlocks | adj | cpu | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | ArrayToBlocks | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | ArrayToBlocks | adj | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | ArrayToBlocks | adj | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | ArrayToBlocks | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | ArrayToBlocks | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | ArrayToBlocks | fwd | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 6.1 |
| smoke | ArrayToBlocks | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | ArrayToBlocks | fwd | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 5.6 |
| smoke | ArrayToBlocks | fwd | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | ArrayToBlocks | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | ArrayToBlocks | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | ArrayToBlocks | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 4.4 |
| smoke | ArrayToBlocks | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | ArrayToBlocks | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 4.2 |
| smoke | ArrayToBlocks | normal | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | ArrayToBlocks | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | ArrayToBlocks | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | BlocksToArray | adj | cpu | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | BlocksToArray | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | BlocksToArray | adj | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | BlocksToArray | adj | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | BlocksToArray | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | BlocksToArray | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | BlocksToArray | fwd | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 4.8 |
| smoke | BlocksToArray | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | BlocksToArray | fwd | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 4.4 |
| smoke | BlocksToArray | fwd | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | BlocksToArray | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | BlocksToArray | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | BlocksToArray | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 4.1 |
| smoke | BlocksToArray | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | BlocksToArray | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 3.8 |
| smoke | BlocksToArray | normal | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | BlocksToArray | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | BlocksToArray | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Chain | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.89 |
| smoke | Chain | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Chain | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.88 |
| smoke | Chain | adj | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.8 |
| smoke | Chain | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Chain | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Chain | fwd | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 0.95 |
| smoke | Chain | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Chain | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.88 |
| smoke | Chain | fwd | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.79 |
| smoke | Chain | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Chain | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Chain | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 0.92 |
| smoke | Chain | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Chain | normal | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.89 |
| smoke | Chain | normal | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.86 |
| smoke | Chain | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Chain | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Concat | adj | cpu | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Concat | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Concat | adj | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Concat | adj | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Concat | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Concat | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Concat | fwd | cpu | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Concat | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Concat | fwd | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Concat | fwd | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Concat | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Concat | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Concat | normal | cpu | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Concat | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Concat | normal | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Concat | normal | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Concat | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Concat | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Dense | adj | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.6 |
| smoke | Dense | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Dense | adj | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.5 |
| smoke | Dense | adj | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 1.3 |
| smoke | Dense | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Dense | adj | cuda | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.2 |
| smoke | Dense | fwd | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.7 |
| smoke | Dense | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Dense | fwd | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.4 |
| smoke | Dense | fwd | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 1.3 |
| smoke | Dense | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Dense | fwd | cuda | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.2 |
| smoke | Dense | normal | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.6 |
| smoke | Dense | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Dense | normal | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.4 |
| smoke | Dense | normal | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 1.4 |
| smoke | Dense | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Dense | normal | cuda | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.2 |
| smoke | DenseSimple | adj | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.6 |
| smoke | DenseSimple | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | DenseSimple | adj | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.4 |
| smoke | DenseSimple | adj | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 1.4 |
| smoke | DenseSimple | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | DenseSimple | adj | cuda | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.2 |
| smoke | DenseSimple | fwd | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.7 |
| smoke | DenseSimple | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | DenseSimple | fwd | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.4 |
| smoke | DenseSimple | fwd | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 1.4 |
| smoke | DenseSimple | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | DenseSimple | fwd | cuda | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.2 |
| smoke | DenseSimple | normal | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.7 |
| smoke | DenseSimple | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | DenseSimple | normal | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.5 |
| smoke | DenseSimple | normal | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 1.4 |
| smoke | DenseSimple | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | DenseSimple | normal | cuda | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.2 |
| smoke | Diagonal | adj | cpu | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.85 |
| smoke | Diagonal | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Diagonal | adj | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.69 |
| smoke | Diagonal | adj | cuda | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.75 |
| smoke | Diagonal | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Diagonal | adj | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.62 |
| smoke | Diagonal | fwd | cpu | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.84 |
| smoke | Diagonal | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Diagonal | fwd | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.69 |
| smoke | Diagonal | fwd | cuda | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.73 |
| smoke | Diagonal | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Diagonal | fwd | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.63 |
| smoke | Diagonal | normal | cpu | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.85 |
| smoke | Diagonal | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Diagonal | normal | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.72 |
| smoke | Diagonal | normal | cuda | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.72 |
| smoke | Diagonal | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Diagonal | normal | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.61 |
| smoke | Identity | adj | cpu | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.7 |
| smoke | Identity | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Identity | adj | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.72 |
| smoke | Identity | adj | cuda | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.73 |
| smoke | Identity | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Identity | adj | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.74 |
| smoke | Identity | fwd | cpu | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.7 |
| smoke | Identity | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Identity | fwd | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.72 |
| smoke | Identity | fwd | cuda | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.72 |
| smoke | Identity | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Identity | fwd | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.76 |
| smoke | Identity | normal | cpu | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.69 |
| smoke | Identity | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Identity | normal | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.74 |
| smoke | Identity | normal | cuda | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.71 |
| smoke | Identity | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Identity | normal | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.71 |
| smoke | Pad | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.56 |
| smoke | Pad | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Pad | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.48 |
| smoke | Pad | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.49 |
| smoke | Pad | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Pad | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.43 |
| smoke | Pad | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.72 |
| smoke | Pad | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Pad | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.64 |
| smoke | Pad | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.67 |
| smoke | Pad | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Pad | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.57 |
| smoke | Pad | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 1.1 |
| smoke | Pad | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Pad | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 0.98 |
| smoke | Pad | normal | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.9 |
| smoke | Pad | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Pad | normal | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.8 |
| smoke | PadDim | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.6 |
| smoke | PadDim | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | PadDim | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.53 |
| smoke | PadDim | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.51 |
| smoke | PadDim | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | PadDim | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.44 |
| smoke | PadDim | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.59 |
| smoke | PadDim | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | PadDim | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.52 |
| smoke | PadDim | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.54 |
| smoke | PadDim | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | PadDim | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.48 |
| smoke | PadDim | normal | cpu | torch.complex64 | wrapper | default | compiles_slower | -1 | 0 | 0.72 |
| smoke | PadDim | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | PadDim | normal | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.75 |
| smoke | PadDim | normal | cuda | torch.complex64 | wrapper | default | compiles_slower | -1 | 0 | 0.77 |
| smoke | PadDim | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | PadDim | normal | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.75 |
| smoke | Rearrange | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.69 |
| smoke | Rearrange | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Rearrange | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.59 |
| smoke | Rearrange | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.53 |
| smoke | Rearrange | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Rearrange | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.48 |
| smoke | Rearrange | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.71 |
| smoke | Rearrange | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Rearrange | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.58 |
| smoke | Rearrange | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.58 |
| smoke | Rearrange | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Rearrange | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.5 |
| smoke | Rearrange | normal | cpu | torch.complex64 | wrapper | default | compiles_slower | -1 | 0 | 0.75 |
| smoke | Rearrange | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Rearrange | normal | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.7 |
| smoke | Rearrange | normal | cuda | torch.complex64 | wrapper | default | compiles_slower | -1 | 0 | 0.74 |
| smoke | Rearrange | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Rearrange | normal | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.75 |
| smoke | Repeat | adj | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 0.91 |
| smoke | Repeat | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Repeat | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.77 |
| smoke | Repeat | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.73 |
| smoke | Repeat | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Repeat | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.63 |
| smoke | Repeat | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.76 |
| smoke | Repeat | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Repeat | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.68 |
| smoke | Repeat | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.64 |
| smoke | Repeat | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Repeat | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.55 |
| smoke | Repeat | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 2.3 |
| smoke | Repeat | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Repeat | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 2 |
| smoke | Repeat | normal | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.97 |
| smoke | Repeat | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Repeat | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Sampling | adj | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.3 |
| smoke | Sampling | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sampling | adj | cpu | torch.float32 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Sampling | adj | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 1.2 |
| smoke | Sampling | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sampling | adj | cuda | torch.float32 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Sampling | fwd | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 2.6 |
| smoke | Sampling | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sampling | fwd | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 2.2 |
| smoke | Sampling | fwd | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 2.9 |
| smoke | Sampling | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sampling | fwd | cuda | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 2.5 |
| smoke | Sampling | normal | cpu | torch.float32 | wrapper | default | works_with_breaks | 1 | 0 | 1.7 |
| smoke | Sampling | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sampling | normal | cpu | torch.float32 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Sampling | normal | cuda | torch.float32 | wrapper | default | works_with_breaks | 1 | 0 | 1.4 |
| smoke | Sampling | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sampling | normal | cuda | torch.float32 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Scalar | adj | cpu | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.88 |
| smoke | Scalar | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Scalar | adj | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.71 |
| smoke | Scalar | adj | cuda | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.7 |
| smoke | Scalar | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Scalar | adj | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.59 |
| smoke | Scalar | fwd | cpu | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.84 |
| smoke | Scalar | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Scalar | fwd | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.72 |
| smoke | Scalar | fwd | cuda | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.72 |
| smoke | Scalar | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Scalar | fwd | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.6 |
| smoke | Scalar | normal | cpu | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.84 |
| smoke | Scalar | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Scalar | normal | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.72 |
| smoke | Scalar | normal | cuda | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.71 |
| smoke | Scalar | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Scalar | normal | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.59 |
| smoke | ShapeSpec | adj | cpu | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.68 |
| smoke | ShapeSpec | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ShapeSpec | adj | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.72 |
| smoke | ShapeSpec | adj | cuda | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.73 |
| smoke | ShapeSpec | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ShapeSpec | adj | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.76 |
| smoke | ShapeSpec | fwd | cpu | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.66 |
| smoke | ShapeSpec | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ShapeSpec | fwd | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.7 |
| smoke | ShapeSpec | fwd | cuda | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.74 |
| smoke | ShapeSpec | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ShapeSpec | fwd | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.74 |
| smoke | ShapeSpec | normal | cpu | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.68 |
| smoke | ShapeSpec | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ShapeSpec | normal | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.67 |
| smoke | ShapeSpec | normal | cuda | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.71 |
| smoke | ShapeSpec | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ShapeSpec | normal | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.68 |
| smoke | SimpleLinop | adj | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.2 |
| smoke | SimpleLinop | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | SimpleLinop | adj | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 0.94 |
| smoke | SimpleLinop | adj | cuda | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.84 |
| smoke | SimpleLinop | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | SimpleLinop | adj | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.7 |
| smoke | SimpleLinop | fwd | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.2 |
| smoke | SimpleLinop | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | SimpleLinop | fwd | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1 |
| smoke | SimpleLinop | fwd | cuda | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.83 |
| smoke | SimpleLinop | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | SimpleLinop | fwd | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.75 |
| smoke | SimpleLinop | normal | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.4 |
| smoke | SimpleLinop | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | SimpleLinop | normal | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.2 |
| smoke | SimpleLinop | normal | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 0.96 |
| smoke | SimpleLinop | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | SimpleLinop | normal | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.85 |
| smoke | Sleep | adj | cpu | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.71 |
| smoke | Sleep | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sleep | adj | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.73 |
| smoke | Sleep | adj | cuda | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.76 |
| smoke | Sleep | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sleep | adj | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.73 |
| smoke | Sleep | fwd | cpu | torch.float32 | wrapper | default | works | -1 | 0 | 0.95 |
| smoke | Sleep | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sleep | fwd | cpu | torch.float32 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Sleep | fwd | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 22 |
| smoke | Sleep | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | Sleep | fwd | cuda | torch.float32 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Stack | adj | cpu | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Stack | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Stack | adj | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Stack | adj | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Stack | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Stack | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Stack | fwd | cpu | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Stack | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Stack | fwd | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Stack | fwd | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| smoke | Stack | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Stack | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Stack | normal | cpu | torch.complex64 | wrapper | default | works | -1 | 0 | 1 |
| smoke | Stack | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Stack | normal | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Stack | normal | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.81 |
| smoke | Stack | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Stack | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | SumReduce | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.77 |
| smoke | SumReduce | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | SumReduce | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.66 |
| smoke | SumReduce | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.62 |
| smoke | SumReduce | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | SumReduce | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.55 |
| smoke | SumReduce | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.89 |
| smoke | SumReduce | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | SumReduce | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.8 |
| smoke | SumReduce | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.79 |
| smoke | SumReduce | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | SumReduce | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.66 |
| smoke | SumReduce | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 2.2 |
| smoke | SumReduce | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | SumReduce | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 2 |
| smoke | SumReduce | normal | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.97 |
| smoke | SumReduce | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | SumReduce | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | ToDevice | adj | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.1 |
| smoke | ToDevice | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ToDevice | adj | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 0.93 |
| smoke | ToDevice | adj | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 0.97 |
| smoke | ToDevice | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ToDevice | adj | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.87 |
| smoke | ToDevice | fwd | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.1 |
| smoke | ToDevice | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ToDevice | fwd | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 0.93 |
| smoke | ToDevice | fwd | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 0.93 |
| smoke | ToDevice | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ToDevice | fwd | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.83 |
| smoke | ToDevice | normal | cpu | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.69 |
| smoke | ToDevice | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ToDevice | normal | cpu | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.73 |
| smoke | ToDevice | normal | cuda | torch.float32 | wrapper | default | compiles_slower | -1 | 0 | 0.77 |
| smoke | ToDevice | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| smoke | ToDevice | normal | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | -1 | 0 | 0.75 |
| smoke | Truncate | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.58 |
| smoke | Truncate | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Truncate | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.51 |
| smoke | Truncate | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.53 |
| smoke | Truncate | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Truncate | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.47 |
| smoke | Truncate | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.61 |
| smoke | Truncate | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Truncate | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.5 |
| smoke | Truncate | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.54 |
| smoke | Truncate | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Truncate | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.47 |
| smoke | Truncate | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 1.6 |
| smoke | Truncate | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Truncate | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 1.3 |
| smoke | Truncate | normal | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.71 |
| smoke | Truncate | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Truncate | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| smoke | Zero | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.53 |
| smoke | Zero | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Zero | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.48 |
| smoke | Zero | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.46 |
| smoke | Zero | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Zero | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.42 |
| smoke | Zero | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.53 |
| smoke | Zero | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Zero | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.47 |
| smoke | Zero | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.47 |
| smoke | Zero | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Zero | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.41 |
| smoke | Zero | normal | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.81 |
| smoke | Zero | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Zero | normal | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.69 |
| smoke | Zero | normal | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.63 |
| smoke | Zero | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| smoke | Zero | normal | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.55 |
| deep | Convolution | adj | cpu | torch.float32 | inner | default | compiles_slower | 0 | 0 | 0.89 |
| deep | Convolution | adj | cpu | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.8 |
| deep | Convolution | adj | cpu | torch.float32 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | adj | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | adj | cpu | torch.float32 | inner | fullgraph | compiles_slower | 0 | 0 | 0.8 |
| deep | Convolution | adj | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.74 |
| deep | Convolution | adj | cuda | torch.float32 | inner | default | compiles_slower | 0 | 0 | 0.73 |
| deep | Convolution | adj | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.68 |
| deep | Convolution | adj | cuda | torch.float32 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | adj | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | adj | cuda | torch.float32 | inner | fullgraph | compiles_slower | 0 | 0 | 0.62 |
| deep | Convolution | adj | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.62 |
| deep | Convolution | fwd | cpu | torch.float32 | inner | default | compiles_slower | 0 | 0 | 0.86 |
| deep | Convolution | fwd | cpu | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.8 |
| deep | Convolution | fwd | cpu | torch.float32 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | fwd | cpu | torch.float32 | inner | fullgraph | compiles_slower | 0 | 0 | 0.75 |
| deep | Convolution | fwd | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.73 |
| deep | Convolution | fwd | cuda | torch.float32 | inner | default | compiles_slower | 0 | 0 | 0.63 |
| deep | Convolution | fwd | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.71 |
| deep | Convolution | fwd | cuda | torch.float32 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | fwd | cuda | torch.float32 | inner | fullgraph | compiles_slower | 0 | 0 | 0.54 |
| deep | Convolution | fwd | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.66 |
| deep | Convolution | normal | cpu | torch.float32 | inner | default | works | 0 | 0 | 1 |
| deep | Convolution | normal | cpu | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.88 |
| deep | Convolution | normal | cpu | torch.float32 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | normal | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | normal | cpu | torch.float32 | inner | fullgraph | works | 0 | 0 | 0.99 |
| deep | Convolution | normal | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.82 |
| deep | Convolution | normal | cuda | torch.float32 | inner | default | works | 0 | 0 | 1.1 |
| deep | Convolution | normal | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.82 |
| deep | Convolution | normal | cuda | torch.float32 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | normal | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Convolution | normal | cuda | torch.float32 | inner | fullgraph | works | 0 | 0 | 0.92 |
| deep | Convolution | normal | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.81 |
| deep | Convolution | adj | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1 |
| deep | Convolution | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.87 |
| deep | Convolution | adj | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | adj | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 0.91 |
| deep | Convolution | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.8 |
| deep | Convolution | adj | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 0.92 |
| deep | Convolution | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.77 |
| deep | Convolution | adj | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | adj | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.78 |
| deep | Convolution | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.73 |
| deep | Convolution | fwd | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1 |
| deep | Convolution | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.85 |
| deep | Convolution | fwd | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | fwd | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 0.91 |
| deep | Convolution | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.84 |
| deep | Convolution | fwd | cuda | torch.float32 | wrapper | default | compiles_slower | 0 | 0 | 0.88 |
| deep | Convolution | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.8 |
| deep | Convolution | fwd | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | fwd | cuda | torch.float32 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.74 |
| deep | Convolution | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.76 |
| deep | Convolution | normal | cpu | torch.float32 | wrapper | default | works | 0 | 0 | 1.3 |
| deep | Convolution | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 0.97 |
| deep | Convolution | normal | cpu | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | normal | cpu | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.2 |
| deep | Convolution | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 0.9 |
| deep | Convolution | normal | cuda | torch.float32 | wrapper | default | works | 0 | 0 | 1.4 |
| deep | Convolution | normal | cuda | torch.complex64 | wrapper | default | works | 0 | 0 | 0.94 |
| deep | Convolution | normal | cuda | torch.float32 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Convolution | normal | cuda | torch.float32 | wrapper | fullgraph | works | 0 | 0 | 1.3 |
| deep | Convolution | normal | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.87 |
| deep | FFT | adj | cpu | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.4 |
| deep | FFT | adj | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FFT | adj | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.37 |
| deep | FFT | adj | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.38 |
| deep | FFT | adj | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FFT | adj | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.39 |
| deep | FFT | fwd | cpu | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.4 |
| deep | FFT | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FFT | fwd | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.38 |
| deep | FFT | fwd | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.39 |
| deep | FFT | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FFT | fwd | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.39 |
| deep | FFT | normal | cpu | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.088 |
| deep | FFT | normal | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FFT | normal | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.018 |
| deep | FFT | normal | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.64 |
| deep | FFT | normal | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FFT | normal | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.25 |
| deep | FFT | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.44 |
| deep | FFT | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FFT | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.41 |
| deep | FFT | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.42 |
| deep | FFT | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FFT | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.44 |
| deep | FFT | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.44 |
| deep | FFT | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FFT | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.42 |
| deep | FFT | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.4 |
| deep | FFT | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FFT | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.41 |
| deep | FFT | normal | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.69 |
| deep | FFT | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FFT | normal | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.7 |
| deep | FFT | normal | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.69 |
| deep | FFT | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FFT | normal | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.73 |
| deep | FastSamplingNUFFT | adj | cpu | torch.complex64 | inner | default | works_with_breaks | 3 | 0 | 0.75 |
| deep | FastSamplingNUFFT | adj | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | adj | cpu | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | FastSamplingNUFFT | adj | cuda | torch.complex64 | inner | default | works_with_breaks | 3 | 0 | 0.67 |
| deep | FastSamplingNUFFT | adj | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | adj | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | FastSamplingNUFFT | fwd | cpu | torch.complex64 | inner | default | works | 0 | 0 | 1.3 |
| deep | FastSamplingNUFFT | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | fwd | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 1.2 |
| deep | FastSamplingNUFFT | fwd | cuda | torch.complex64 | inner | default | works | 0 | 0 | 1.7 |
| deep | FastSamplingNUFFT | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | fwd | cuda | torch.complex64 | inner | fullgraph | works | 0 | 0 | 1.5 |
| deep | FastSamplingNUFFT | normal | cpu | torch.complex64 | inner | default | works_with_breaks | 4 | 0 | 0.94 |
| deep | FastSamplingNUFFT | normal | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | normal | cpu | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | FastSamplingNUFFT | normal | cuda | torch.complex64 | inner | default | works_with_breaks | 4 | 0 | 0.85 |
| deep | FastSamplingNUFFT | normal | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | normal | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | FastSamplingNUFFT | adj | cpu | torch.complex64 | wrapper | default | works_with_breaks | 3 | 0 | 0.75 |
| deep | FastSamplingNUFFT | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | adj | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | FastSamplingNUFFT | adj | cuda | torch.complex64 | wrapper | default | works_with_breaks | 4 | 0 | 0.63 |
| deep | FastSamplingNUFFT | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | FastSamplingNUFFT | fwd | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 1.3 |
| deep | FastSamplingNUFFT | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | fwd | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 1.3 |
| deep | FastSamplingNUFFT | fwd | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 1.4 |
| deep | FastSamplingNUFFT | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | FastSamplingNUFFT | normal | cpu | torch.complex64 | wrapper | default | works_with_breaks | 6 | 0 | 0.81 |
| deep | FastSamplingNUFFT | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | normal | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | FastSamplingNUFFT | normal | cuda | torch.complex64 | wrapper | default | works_with_breaks | 7 | 0 | 0.69 |
| deep | FastSamplingNUFFT | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | FastSamplingNUFFT | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | Interpolate | adj | cpu | torch.complex64 | inner | default | works | 0 | 0 | 3.6 |
| deep | Interpolate | adj | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Interpolate | adj | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 3.4 |
| deep | Interpolate | adj | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | Interpolate | adj | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Interpolate | adj | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | Interpolate | fwd | cpu | torch.complex64 | inner | default | works | 0 | 0 | 3.8 |
| deep | Interpolate | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Interpolate | fwd | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 3.6 |
| deep | Interpolate | fwd | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | Interpolate | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Interpolate | fwd | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | Interpolate | normal | cpu | torch.complex64 | inner | default | works | 0 | 0 | 4.5 |
| deep | Interpolate | normal | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Interpolate | normal | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 4.2 |
| deep | Interpolate | normal | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | Interpolate | normal | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | Interpolate | normal | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | Interpolate | adj | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 3.6 |
| deep | Interpolate | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Interpolate | adj | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 3.4 |
| deep | Interpolate | adj | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| deep | Interpolate | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Interpolate | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | Interpolate | fwd | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 3.9 |
| deep | Interpolate | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Interpolate | fwd | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 3.5 |
| deep | Interpolate | fwd | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| deep | Interpolate | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Interpolate | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | Interpolate | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 4.6 |
| deep | Interpolate | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Interpolate | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 4.4 |
| deep | Interpolate | normal | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| deep | Interpolate | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | Interpolate | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | NUFFT | adj | cpu | torch.complex64 | inner | default | works | 0 | 0 | 1.3 |
| deep | NUFFT | adj | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | NUFFT | adj | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 1.3 |
| deep | NUFFT | adj | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | NUFFT | adj | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | NUFFT | adj | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | NUFFT | fwd | cpu | torch.complex64 | inner | default | works | 0 | 0 | 1.3 |
| deep | NUFFT | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | NUFFT | fwd | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 1.3 |
| deep | NUFFT | fwd | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | NUFFT | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | NUFFT | fwd | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | NUFFT | normal | cpu | torch.complex64 | inner | default | works | 0 | 0 | 1.4 |
| deep | NUFFT | normal | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | NUFFT | normal | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 1.3 |
| deep | NUFFT | normal | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | NUFFT | normal | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | NUFFT | normal | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | NUFFT | adj | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 1.4 |
| deep | NUFFT | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | NUFFT | adj | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 1.3 |
| deep | NUFFT | adj | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| deep | NUFFT | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | NUFFT | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | NUFFT | fwd | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 1.4 |
| deep | NUFFT | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | NUFFT | fwd | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 1.3 |
| deep | NUFFT | fwd | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| deep | NUFFT | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | NUFFT | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | NUFFT | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 1.4 |
| deep | NUFFT | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | NUFFT | normal | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 1.4 |
| deep | NUFFT | normal | cuda | torch.complex64 | wrapper | default | fails_loud | 0 | - | - |
| deep | NUFFT | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | NUFFT | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | SamplingNUFFT | adj | cpu | torch.complex64 | inner | default | works_with_breaks | 1 | 0 | 0.75 |
| deep | SamplingNUFFT | adj | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | adj | cpu | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | SamplingNUFFT | adj | cuda | torch.complex64 | inner | default | works_with_breaks | 1 | 0 | 0.6 |
| deep | SamplingNUFFT | adj | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | adj | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | SamplingNUFFT | fwd | cpu | torch.complex64 | inner | default | works | 0 | 0 | 0.92 |
| deep | SamplingNUFFT | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | fwd | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.89 |
| deep | SamplingNUFFT | fwd | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.72 |
| deep | SamplingNUFFT | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | fwd | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.73 |
| deep | SamplingNUFFT | normal | cpu | torch.complex64 | inner | default | works_with_breaks | 2 | 0 | 0.83 |
| deep | SamplingNUFFT | normal | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | normal | cpu | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | SamplingNUFFT | normal | cuda | torch.complex64 | inner | default | works_with_breaks | 2 | 0 | 0.64 |
| deep | SamplingNUFFT | normal | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | normal | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | SamplingNUFFT | adj | cpu | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.81 |
| deep | SamplingNUFFT | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | adj | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | SamplingNUFFT | adj | cuda | torch.complex64 | wrapper | default | works_with_breaks | 2 | 0 | 0.63 |
| deep | SamplingNUFFT | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | adj | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | SamplingNUFFT | fwd | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 0.98 |
| deep | SamplingNUFFT | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | fwd | cpu | torch.complex64 | wrapper | fullgraph | works | 0 | 0 | 0.96 |
| deep | SamplingNUFFT | fwd | cuda | torch.complex64 | wrapper | default | works_with_breaks | 1 | 0 | 0.74 |
| deep | SamplingNUFFT | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | fwd | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | SamplingNUFFT | normal | cpu | torch.complex64 | wrapper | default | works_with_breaks | 5 | 0 | 0.71 |
| deep | SamplingNUFFT | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | normal | cpu | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | SamplingNUFFT | normal | cuda | torch.complex64 | wrapper | default | works_with_breaks | 6 | 0 | 0.56 |
| deep | SamplingNUFFT | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | SamplingNUFFT | normal | cuda | torch.complex64 | wrapper | fullgraph | fails_loud | 0 | - | - |
| deep | ToeplitzPSF | adj | cpu | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.52 |
| deep | ToeplitzPSF | adj | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | adj | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.48 |
| deep | ToeplitzPSF | adj | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.53 |
| deep | ToeplitzPSF | adj | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | adj | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.47 |
| deep | ToeplitzPSF | fwd | cpu | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.52 |
| deep | ToeplitzPSF | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | fwd | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.47 |
| deep | ToeplitzPSF | fwd | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.51 |
| deep | ToeplitzPSF | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | fwd | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.46 |
| deep | ToeplitzPSF | normal | cpu | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.78 |
| deep | ToeplitzPSF | normal | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | normal | cpu | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.73 |
| deep | ToeplitzPSF | normal | cuda | torch.complex64 | inner | default | compiles_slower | 0 | 0 | 0.75 |
| deep | ToeplitzPSF | normal | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | normal | cuda | torch.complex64 | inner | fullgraph | compiles_slower | 0 | 0 | 0.68 |
| deep | ToeplitzPSF | adj | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.73 |
| deep | ToeplitzPSF | adj | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | adj | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.67 |
| deep | ToeplitzPSF | adj | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.66 |
| deep | ToeplitzPSF | adj | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | adj | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.66 |
| deep | ToeplitzPSF | fwd | cpu | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.8 |
| deep | ToeplitzPSF | fwd | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | fwd | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.72 |
| deep | ToeplitzPSF | fwd | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.7 |
| deep | ToeplitzPSF | fwd | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | fwd | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.62 |
| deep | ToeplitzPSF | normal | cpu | torch.complex64 | wrapper | default | works | 0 | 0 | 0.91 |
| deep | ToeplitzPSF | normal | cpu | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | normal | cpu | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.81 |
| deep | ToeplitzPSF | normal | cuda | torch.complex64 | wrapper | default | compiles_slower | 0 | 0 | 0.78 |
| deep | ToeplitzPSF | normal | cuda | torch.complex64 | wrapper | eager | works | 0 | 0 | - |
| deep | ToeplitzPSF | normal | cuda | torch.complex64 | wrapper | fullgraph | compiles_slower | 0 | 0 | 0.65 |
| deep | functional.fold | fwd | cpu | torch.complex64 | inner | default | works | 0 | 0 | 5.2 |
| deep | functional.fold | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | functional.fold | fwd | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 4.7 |
| deep | functional.fold | fwd | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | functional.fold | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | functional.fold | fwd | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | functional.grid | fwd | cpu | torch.complex64 | inner | default | works | 0 | 0 | 2.3 |
| deep | functional.grid | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | functional.grid | fwd | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 2.2 |
| deep | functional.grid | fwd | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | functional.grid | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | functional.grid | fwd | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | functional.unfold | fwd | cpu | torch.complex64 | inner | default | works | 0 | 0 | 5.3 |
| deep | functional.unfold | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | functional.unfold | fwd | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 4.7 |
| deep | functional.unfold | fwd | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | functional.unfold | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | functional.unfold | fwd | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |
| deep | functional.ungrid | fwd | cpu | torch.complex64 | inner | default | works | 0 | 0 | 2.5 |
| deep | functional.ungrid | fwd | cpu | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | functional.ungrid | fwd | cpu | torch.complex64 | inner | fullgraph | works | 0 | 0 | 2.4 |
| deep | functional.ungrid | fwd | cuda | torch.complex64 | inner | default | fails_loud | 0 | - | - |
| deep | functional.ungrid | fwd | cuda | torch.complex64 | inner | eager | works | 0 | 0 | - |
| deep | functional.ungrid | fwd | cuda | torch.complex64 | inner | fullgraph | fails_loud | 0 | - | - |

## Speedup table (default config, sorted)

| op | level | method | device | dtype | speedup | class |
|---|---|---|---|---|---|---|
| Sleep | wrapper | fwd | cuda | torch.float32 | 22 | works |
| ArrayToBlocks | wrapper | fwd | cpu | torch.complex64 | 6.1 | works |
| ArrayToBlocks | wrapper | fwd | cpu | torch.complex64 | 5.6 | works |
| functional.unfold | inner | fwd | cpu | torch.complex64 | 5.3 | works |
| functional.fold | inner | fwd | cpu | torch.complex64 | 5.2 | works |
| BlocksToArray | wrapper | fwd | cpu | torch.complex64 | 4.8 | works |
| functional.fold | inner | fwd | cpu | torch.complex64 | 4.7 | works |
| functional.unfold | inner | fwd | cpu | torch.complex64 | 4.7 | works |
| Interpolate | wrapper | normal | cpu | torch.complex64 | 4.6 | works |
| Interpolate | inner | normal | cpu | torch.complex64 | 4.5 | works |
| ArrayToBlocks | wrapper | normal | cpu | torch.complex64 | 4.4 | works |
| BlocksToArray | wrapper | fwd | cpu | torch.complex64 | 4.4 | works |
| Interpolate | wrapper | normal | cpu | torch.complex64 | 4.4 | works |
| Interpolate | inner | normal | cpu | torch.complex64 | 4.2 | works |
| ArrayToBlocks | wrapper | normal | cpu | torch.complex64 | 4.2 | works |
| BlocksToArray | wrapper | normal | cpu | torch.complex64 | 4.1 | works |
| Interpolate | wrapper | fwd | cpu | torch.complex64 | 3.9 | works |
| BlocksToArray | wrapper | normal | cpu | torch.complex64 | 3.8 | works |
| Interpolate | inner | fwd | cpu | torch.complex64 | 3.8 | works |
| Interpolate | inner | adj | cpu | torch.complex64 | 3.6 | works |
| Interpolate | wrapper | adj | cpu | torch.complex64 | 3.6 | works |
| Interpolate | inner | fwd | cpu | torch.complex64 | 3.6 | works |
| Interpolate | wrapper | fwd | cpu | torch.complex64 | 3.5 | works |
| Interpolate | wrapper | adj | cpu | torch.complex64 | 3.4 | works |
| Interpolate | inner | adj | cpu | torch.complex64 | 3.4 | works |
| Sampling | wrapper | fwd | cuda | torch.float32 | 2.9 | works |
| Sampling | wrapper | fwd | cpu | torch.float32 | 2.6 | works |
| Sampling | wrapper | fwd | cuda | torch.float32 | 2.5 | works |
| functional.ungrid | inner | fwd | cpu | torch.complex64 | 2.5 | works |
| functional.ungrid | inner | fwd | cpu | torch.complex64 | 2.4 | works |
| functional.grid | inner | fwd | cpu | torch.complex64 | 2.3 | works |
| Repeat | wrapper | normal | cpu | torch.complex64 | 2.3 | works |
| Sampling | wrapper | fwd | cpu | torch.float32 | 2.2 | works |
| SumReduce | wrapper | normal | cpu | torch.complex64 | 2.2 | works |
| functional.grid | inner | fwd | cpu | torch.complex64 | 2.2 | works |
| Repeat | wrapper | normal | cpu | torch.complex64 | 2 | works |
| SumReduce | wrapper | normal | cpu | torch.complex64 | 2 | works |
| DenseSimple | wrapper | normal | cpu | torch.float32 | 1.7 | works |
| Sampling | wrapper | normal | cpu | torch.float32 | 1.7 | works_with_breaks |
| FastSamplingNUFFT | inner | fwd | cuda | torch.complex64 | 1.7 | works |
| DenseSimple | wrapper | fwd | cpu | torch.float32 | 1.7 | works |
| Dense | wrapper | fwd | cpu | torch.float32 | 1.7 | works |
| Dense | wrapper | normal | cpu | torch.float32 | 1.6 | works |
| DenseSimple | wrapper | adj | cpu | torch.float32 | 1.6 | works |
| Dense | wrapper | adj | cpu | torch.float32 | 1.6 | works |
| Truncate | wrapper | normal | cpu | torch.complex64 | 1.6 | works |
| DenseSimple | wrapper | normal | cpu | torch.float32 | 1.5 | works |
| FastSamplingNUFFT | inner | fwd | cuda | torch.complex64 | 1.5 | works |
| Dense | wrapper | adj | cpu | torch.float32 | 1.5 | works |
| Sampling | wrapper | normal | cuda | torch.float32 | 1.4 | works_with_breaks |
| DenseSimple | wrapper | adj | cpu | torch.float32 | 1.4 | works |
| DenseSimple | wrapper | fwd | cuda | torch.float32 | 1.4 | works |
| Convolution | wrapper | normal | cuda | torch.float32 | 1.4 | works |
| SimpleLinop | wrapper | normal | cpu | torch.float32 | 1.4 | works |
| DenseSimple | wrapper | fwd | cpu | torch.float32 | 1.4 | works |
| Dense | wrapper | fwd | cpu | torch.float32 | 1.4 | works |
| Dense | wrapper | normal | cuda | torch.float32 | 1.4 | works |
| DenseSimple | wrapper | normal | cuda | torch.float32 | 1.4 | works |
| Dense | wrapper | normal | cpu | torch.float32 | 1.4 | works |
| NUFFT | wrapper | adj | cpu | torch.complex64 | 1.4 | works |
| NUFFT | wrapper | fwd | cpu | torch.complex64 | 1.4 | works |
| DenseSimple | wrapper | adj | cuda | torch.float32 | 1.4 | works |
| NUFFT | wrapper | normal | cpu | torch.complex64 | 1.4 | works |
| NUFFT | inner | normal | cpu | torch.complex64 | 1.4 | works |
| NUFFT | wrapper | normal | cpu | torch.complex64 | 1.4 | works |
| FastSamplingNUFFT | wrapper | fwd | cuda | torch.complex64 | 1.4 | works_with_breaks |
| NUFFT | wrapper | adj | cpu | torch.complex64 | 1.3 | works |
| Truncate | wrapper | normal | cpu | torch.complex64 | 1.3 | works |
| NUFFT | inner | normal | cpu | torch.complex64 | 1.3 | works |
| NUFFT | wrapper | fwd | cpu | torch.complex64 | 1.3 | works |
| Sampling | wrapper | adj | cpu | torch.float32 | 1.3 | works |
| NUFFT | inner | fwd | cpu | torch.complex64 | 1.3 | works |
| FastSamplingNUFFT | wrapper | fwd | cpu | torch.complex64 | 1.3 | works |
| FastSamplingNUFFT | wrapper | fwd | cpu | torch.complex64 | 1.3 | works |
| Dense | wrapper | adj | cuda | torch.float32 | 1.3 | works |
| NUFFT | inner | fwd | cpu | torch.complex64 | 1.3 | works |
| Dense | wrapper | fwd | cuda | torch.float32 | 1.3 | works |
| NUFFT | inner | adj | cpu | torch.complex64 | 1.3 | works |
| Convolution | wrapper | normal | cpu | torch.float32 | 1.3 | works |
| Convolution | wrapper | normal | cuda | torch.float32 | 1.3 | works |
| NUFFT | inner | adj | cpu | torch.complex64 | 1.3 | works |
| FastSamplingNUFFT | inner | fwd | cpu | torch.complex64 | 1.3 | works |
| FastSamplingNUFFT | inner | fwd | cpu | torch.complex64 | 1.2 | works |
| DenseSimple | wrapper | fwd | cuda | torch.float32 | 1.2 | works |
| SimpleLinop | wrapper | normal | cpu | torch.float32 | 1.2 | works |
| Dense | wrapper | fwd | cuda | torch.float32 | 1.2 | works |
| SimpleLinop | wrapper | fwd | cpu | torch.float32 | 1.2 | works |
| Convolution | wrapper | normal | cpu | torch.float32 | 1.2 | works |
| Dense | wrapper | normal | cuda | torch.float32 | 1.2 | works |
| DenseSimple | wrapper | normal | cuda | torch.float32 | 1.2 | works |
| SimpleLinop | wrapper | adj | cpu | torch.float32 | 1.2 | works |
| DenseSimple | wrapper | adj | cuda | torch.float32 | 1.2 | works |
| Sampling | wrapper | adj | cuda | torch.float32 | 1.2 | works |
| Dense | wrapper | adj | cuda | torch.float32 | 1.2 | works |
| ToDevice | wrapper | adj | cpu | torch.float32 | 1.1 | works |
| Add | wrapper | normal | cpu | torch.complex64 | 1.1 | works |
| Add | wrapper | normal | cpu | torch.complex64 | 1.1 | works |
| Pad | wrapper | normal | cpu | torch.complex64 | 1.1 | works |
| ToDevice | wrapper | fwd | cpu | torch.float32 | 1.1 | works |
| Convolution | inner | normal | cuda | torch.float32 | 1.1 | works |
| Convolution | inner | normal | cpu | torch.float32 | 1 | works |
| Convolution | wrapper | fwd | cpu | torch.float32 | 1 | works |
| Convolution | wrapper | adj | cpu | torch.float32 | 1 | works |
| Stack | wrapper | normal | cpu | torch.complex64 | 1 | works |
| SimpleLinop | wrapper | fwd | cpu | torch.float32 | 1 | works |
| Convolution | inner | normal | cpu | torch.float32 | 0.99 | works |
| SamplingNUFFT | wrapper | fwd | cpu | torch.complex64 | 0.98 | works |
| Pad | wrapper | normal | cpu | torch.complex64 | 0.98 | works |
| Repeat | wrapper | normal | cuda | torch.complex64 | 0.97 | works_with_breaks |
| ToDevice | wrapper | adj | cuda | torch.float32 | 0.97 | works |
| Convolution | wrapper | normal | cpu | torch.complex64 | 0.97 | works |
| SumReduce | wrapper | normal | cuda | torch.complex64 | 0.97 | works_with_breaks |
| SamplingNUFFT | wrapper | fwd | cpu | torch.complex64 | 0.96 | works |
| SimpleLinop | wrapper | normal | cuda | torch.float32 | 0.96 | works |
| Sleep | wrapper | fwd | cpu | torch.float32 | 0.95 | works |
| Chain | wrapper | fwd | cpu | torch.complex64 | 0.95 | works |
| FastSamplingNUFFT | inner | normal | cpu | torch.complex64 | 0.94 | works_with_breaks |
| SimpleLinop | wrapper | adj | cpu | torch.float32 | 0.94 | works |
| Convolution | wrapper | normal | cuda | torch.complex64 | 0.94 | works |
| ToDevice | wrapper | fwd | cpu | torch.float32 | 0.93 | works |
| ToDevice | wrapper | fwd | cuda | torch.float32 | 0.93 | works |
| ToDevice | wrapper | adj | cpu | torch.float32 | 0.93 | works |
| SamplingNUFFT | inner | fwd | cpu | torch.complex64 | 0.92 | works |
| Convolution | inner | normal | cuda | torch.float32 | 0.92 | works |
| Chain | wrapper | normal | cpu | torch.complex64 | 0.92 | works |
| Convolution | wrapper | adj | cuda | torch.float32 | 0.92 | works |
| Convolution | wrapper | fwd | cpu | torch.float32 | 0.91 | works |
| Convolution | wrapper | adj | cpu | torch.float32 | 0.91 | works |
| Repeat | wrapper | adj | cpu | torch.complex64 | 0.91 | works |
| ToeplitzPSF | wrapper | normal | cpu | torch.complex64 | 0.91 | works |
| Convolution | wrapper | normal | cpu | torch.complex64 | 0.9 | works |
| Pad | wrapper | normal | cuda | torch.complex64 | 0.9 | compiles_slower |
| Chain | wrapper | normal | cpu | torch.complex64 | 0.89 | compiles_slower |
| SamplingNUFFT | inner | fwd | cpu | torch.complex64 | 0.89 | compiles_slower |
| SumReduce | wrapper | fwd | cpu | torch.complex64 | 0.89 | compiles_slower |
| Convolution | inner | adj | cpu | torch.float32 | 0.89 | compiles_slower |
| Chain | wrapper | adj | cpu | torch.complex64 | 0.89 | compiles_slower |
| Convolution | inner | normal | cpu | torch.complex64 | 0.88 | compiles_slower |
| Chain | wrapper | fwd | cpu | torch.complex64 | 0.88 | compiles_slower |
| Chain | wrapper | adj | cpu | torch.complex64 | 0.88 | compiles_slower |
| Convolution | wrapper | fwd | cuda | torch.float32 | 0.88 | compiles_slower |
| Scalar | wrapper | adj | cpu | torch.float32 | 0.88 | compiles_slower |
| Convolution | wrapper | normal | cuda | torch.complex64 | 0.87 | compiles_slower |
| Convolution | wrapper | adj | cpu | torch.complex64 | 0.87 | compiles_slower |
| ToDevice | wrapper | adj | cuda | torch.float32 | 0.87 | compiles_slower |
| Chain | wrapper | normal | cuda | torch.complex64 | 0.86 | works_with_breaks |
| Convolution | inner | fwd | cpu | torch.float32 | 0.86 | compiles_slower |
| SimpleLinop | wrapper | normal | cuda | torch.float32 | 0.85 | compiles_slower |
| Diagonal | wrapper | normal | cpu | torch.float32 | 0.85 | compiles_slower |
| Convolution | wrapper | fwd | cpu | torch.complex64 | 0.85 | compiles_slower |
| FastSamplingNUFFT | inner | normal | cuda | torch.complex64 | 0.85 | works_with_breaks |
| Diagonal | wrapper | adj | cpu | torch.float32 | 0.85 | compiles_slower |
| SimpleLinop | wrapper | adj | cuda | torch.float32 | 0.84 | compiles_slower |
| Scalar | wrapper | normal | cpu | torch.float32 | 0.84 | compiles_slower |
| Convolution | wrapper | fwd | cpu | torch.complex64 | 0.84 | compiles_slower |
| Scalar | wrapper | fwd | cpu | torch.float32 | 0.84 | compiles_slower |
| Diagonal | wrapper | fwd | cpu | torch.float32 | 0.84 | compiles_slower |
| SamplingNUFFT | inner | normal | cpu | torch.complex64 | 0.83 | works_with_breaks |
| ToDevice | wrapper | fwd | cuda | torch.float32 | 0.83 | compiles_slower |
| SimpleLinop | wrapper | fwd | cuda | torch.float32 | 0.83 | compiles_slower |
| Convolution | inner | normal | cuda | torch.complex64 | 0.82 | compiles_slower |
| Convolution | inner | normal | cpu | torch.complex64 | 0.82 | compiles_slower |
| Convolution | inner | normal | cuda | torch.complex64 | 0.81 | compiles_slower |
| SamplingNUFFT | wrapper | adj | cpu | torch.complex64 | 0.81 | works_with_breaks |
| Zero | wrapper | normal | cpu | torch.complex64 | 0.81 | compiles_slower |
| ToeplitzPSF | wrapper | normal | cpu | torch.complex64 | 0.81 | compiles_slower |
| Add | wrapper | fwd | cuda | torch.complex64 | 0.81 | works_with_breaks |
| Stack | wrapper | normal | cuda | torch.complex64 | 0.81 | compiles_slower |
| FastSamplingNUFFT | wrapper | normal | cpu | torch.complex64 | 0.81 | works_with_breaks |
| Convolution | wrapper | adj | cpu | torch.complex64 | 0.8 | compiles_slower |
| Convolution | inner | fwd | cpu | torch.complex64 | 0.8 | compiles_slower |
| Pad | wrapper | normal | cuda | torch.complex64 | 0.8 | compiles_slower |
| Chain | wrapper | adj | cuda | torch.complex64 | 0.8 | works_with_breaks |
| ToeplitzPSF | wrapper | fwd | cpu | torch.complex64 | 0.8 | compiles_slower |
| SumReduce | wrapper | fwd | cpu | torch.complex64 | 0.8 | compiles_slower |
| Convolution | wrapper | fwd | cuda | torch.complex64 | 0.8 | compiles_slower |
| Convolution | inner | adj | cpu | torch.complex64 | 0.8 | compiles_slower |
| Convolution | inner | adj | cpu | torch.float32 | 0.8 | compiles_slower |
| SumReduce | wrapper | fwd | cuda | torch.complex64 | 0.79 | compiles_slower |
| Chain | wrapper | fwd | cuda | torch.complex64 | 0.79 | works_with_breaks |
| Add | wrapper | adj | cuda | torch.complex64 | 0.79 | works_with_breaks |
| Convolution | wrapper | adj | cuda | torch.float32 | 0.78 | compiles_slower |
| ToeplitzPSF | inner | normal | cpu | torch.complex64 | 0.78 | compiles_slower |
| ToeplitzPSF | wrapper | normal | cuda | torch.complex64 | 0.78 | compiles_slower |
| Add | wrapper | adj | cpu | torch.complex64 | 0.78 | compiles_slower |
| Add | wrapper | fwd | cpu | torch.complex64 | 0.78 | compiles_slower |
| ToDevice | wrapper | normal | cuda | torch.float32 | 0.77 | compiles_slower |
| PadDim | wrapper | normal | cuda | torch.complex64 | 0.77 | compiles_slower |
| Convolution | wrapper | adj | cuda | torch.complex64 | 0.77 | compiles_slower |
| Repeat | wrapper | adj | cpu | torch.complex64 | 0.77 | compiles_slower |
| SumReduce | wrapper | adj | cpu | torch.complex64 | 0.77 | compiles_slower |
| Repeat | wrapper | fwd | cpu | torch.complex64 | 0.76 | compiles_slower |
| Identity | wrapper | fwd | cuda | torch.float32 | 0.76 | compiles_slower |
| ShapeSpec | wrapper | adj | cuda | torch.float32 | 0.76 | compiles_slower |
| Convolution | wrapper | fwd | cuda | torch.complex64 | 0.76 | compiles_slower |
| Sleep | wrapper | adj | cuda | torch.float32 | 0.76 | compiles_slower |
| Rearrange | wrapper | normal | cuda | torch.complex64 | 0.75 | compiles_slower |
| Rearrange | wrapper | normal | cpu | torch.complex64 | 0.75 | compiles_slower |
| Add | wrapper | fwd | cpu | torch.complex64 | 0.75 | compiles_slower |
| ToDevice | wrapper | normal | cuda | torch.float32 | 0.75 | compiles_slower |
| PadDim | wrapper | normal | cpu | torch.complex64 | 0.75 | compiles_slower |
| Diagonal | wrapper | adj | cuda | torch.float32 | 0.75 | compiles_slower |
| FastSamplingNUFFT | inner | adj | cpu | torch.complex64 | 0.75 | works_with_breaks |
| SimpleLinop | wrapper | fwd | cuda | torch.float32 | 0.75 | compiles_slower |
| Convolution | inner | fwd | cpu | torch.float32 | 0.75 | compiles_slower |
| FastSamplingNUFFT | wrapper | adj | cpu | torch.complex64 | 0.75 | works_with_breaks |
| PadDim | wrapper | normal | cuda | torch.complex64 | 0.75 | compiles_slower |
| SamplingNUFFT | inner | adj | cpu | torch.complex64 | 0.75 | works_with_breaks |
| ToeplitzPSF | inner | normal | cuda | torch.complex64 | 0.75 | compiles_slower |
| Identity | wrapper | adj | cuda | torch.float32 | 0.74 | compiles_slower |
| Identity | wrapper | normal | cpu | torch.float32 | 0.74 | compiles_slower |
| ShapeSpec | wrapper | fwd | cuda | torch.float32 | 0.74 | compiles_slower |
| SamplingNUFFT | wrapper | fwd | cuda | torch.complex64 | 0.74 | works_with_breaks |
| ShapeSpec | wrapper | fwd | cuda | torch.float32 | 0.74 | compiles_slower |
| Convolution | wrapper | fwd | cuda | torch.float32 | 0.74 | compiles_slower |
| Rearrange | wrapper | normal | cuda | torch.complex64 | 0.74 | compiles_slower |
| Convolution | inner | adj | cpu | torch.complex64 | 0.74 | compiles_slower |
| ToDevice | wrapper | normal | cpu | torch.float32 | 0.73 | compiles_slower |
| FFT | wrapper | normal | cuda | torch.complex64 | 0.73 | compiles_slower |
| Convolution | wrapper | adj | cuda | torch.complex64 | 0.73 | compiles_slower |
| Sleep | wrapper | adj | cuda | torch.float32 | 0.73 | compiles_slower |
| ToeplitzPSF | wrapper | adj | cpu | torch.complex64 | 0.73 | compiles_slower |
| Add | wrapper | normal | cuda | torch.complex64 | 0.73 | works_with_breaks |
| Repeat | wrapper | adj | cuda | torch.complex64 | 0.73 | compiles_slower |
| ToeplitzPSF | inner | normal | cpu | torch.complex64 | 0.73 | compiles_slower |
| Sleep | wrapper | adj | cpu | torch.float32 | 0.73 | compiles_slower |
| Convolution | inner | fwd | cpu | torch.complex64 | 0.73 | compiles_slower |
| Convolution | inner | adj | cuda | torch.float32 | 0.73 | compiles_slower |
| Diagonal | wrapper | fwd | cuda | torch.float32 | 0.73 | compiles_slower |
| Identity | wrapper | adj | cuda | torch.float32 | 0.73 | compiles_slower |
| ShapeSpec | wrapper | adj | cuda | torch.float32 | 0.73 | compiles_slower |
| SamplingNUFFT | inner | fwd | cuda | torch.complex64 | 0.73 | compiles_slower |
| ToeplitzPSF | wrapper | fwd | cpu | torch.complex64 | 0.72 | compiles_slower |
| Identity | wrapper | fwd | cuda | torch.float32 | 0.72 | compiles_slower |
| Scalar | wrapper | normal | cpu | torch.float32 | 0.72 | compiles_slower |
| Scalar | wrapper | fwd | cuda | torch.float32 | 0.72 | compiles_slower |
| Identity | wrapper | adj | cpu | torch.float32 | 0.72 | compiles_slower |
| Pad | wrapper | fwd | cpu | torch.complex64 | 0.72 | compiles_slower |
| Identity | wrapper | fwd | cpu | torch.float32 | 0.72 | compiles_slower |
| SamplingNUFFT | inner | fwd | cuda | torch.complex64 | 0.72 | compiles_slower |
| PadDim | wrapper | normal | cpu | torch.complex64 | 0.72 | compiles_slower |
| Scalar | wrapper | fwd | cpu | torch.float32 | 0.72 | compiles_slower |
| Diagonal | wrapper | normal | cpu | torch.float32 | 0.72 | compiles_slower |
| ShapeSpec | wrapper | adj | cpu | torch.float32 | 0.72 | compiles_slower |
| Diagonal | wrapper | normal | cuda | torch.float32 | 0.72 | compiles_slower |
| Identity | wrapper | normal | cuda | torch.float32 | 0.71 | compiles_slower |
| Identity | wrapper | normal | cuda | torch.float32 | 0.71 | compiles_slower |
| Sleep | wrapper | adj | cpu | torch.float32 | 0.71 | compiles_slower |
| Convolution | inner | fwd | cuda | torch.complex64 | 0.71 | compiles_slower |
| Scalar | wrapper | normal | cuda | torch.float32 | 0.71 | compiles_slower |
| Rearrange | wrapper | fwd | cpu | torch.complex64 | 0.71 | compiles_slower |
| SamplingNUFFT | wrapper | normal | cpu | torch.complex64 | 0.71 | works_with_breaks |
| Truncate | wrapper | normal | cuda | torch.complex64 | 0.71 | works_with_breaks |
| Scalar | wrapper | adj | cpu | torch.float32 | 0.71 | compiles_slower |
| Add | wrapper | adj | cpu | torch.complex64 | 0.71 | compiles_slower |
| ShapeSpec | wrapper | normal | cuda | torch.float32 | 0.71 | compiles_slower |
| SimpleLinop | wrapper | adj | cuda | torch.float32 | 0.7 | compiles_slower |
| ToeplitzPSF | wrapper | fwd | cuda | torch.complex64 | 0.7 | compiles_slower |
| Scalar | wrapper | adj | cuda | torch.float32 | 0.7 | compiles_slower |
| Rearrange | wrapper | normal | cpu | torch.complex64 | 0.7 | compiles_slower |
| Identity | wrapper | adj | cpu | torch.float32 | 0.7 | compiles_slower |
| ShapeSpec | wrapper | fwd | cpu | torch.float32 | 0.7 | compiles_slower |
| Identity | wrapper | fwd | cpu | torch.float32 | 0.7 | compiles_slower |
| FFT | wrapper | normal | cpu | torch.complex64 | 0.7 | compiles_slower |
| Diagonal | wrapper | adj | cpu | torch.float32 | 0.69 | compiles_slower |
| Rearrange | wrapper | adj | cpu | torch.complex64 | 0.69 | compiles_slower |
| Zero | wrapper | normal | cpu | torch.complex64 | 0.69 | compiles_slower |
| ToDevice | wrapper | normal | cpu | torch.float32 | 0.69 | compiles_slower |
| Identity | wrapper | normal | cpu | torch.float32 | 0.69 | compiles_slower |
| Diagonal | wrapper | fwd | cpu | torch.float32 | 0.69 | compiles_slower |
| FastSamplingNUFFT | wrapper | normal | cuda | torch.complex64 | 0.69 | works_with_breaks |
| FFT | wrapper | normal | cuda | torch.complex64 | 0.69 | compiles_slower |
| FFT | wrapper | normal | cpu | torch.complex64 | 0.69 | compiles_slower |
| ShapeSpec | wrapper | normal | cuda | torch.float32 | 0.68 | compiles_slower |
| Repeat | wrapper | fwd | cpu | torch.complex64 | 0.68 | compiles_slower |
| ToeplitzPSF | inner | normal | cuda | torch.complex64 | 0.68 | compiles_slower |
| ShapeSpec | wrapper | adj | cpu | torch.float32 | 0.68 | compiles_slower |
| Convolution | inner | adj | cuda | torch.complex64 | 0.68 | compiles_slower |
| ShapeSpec | wrapper | normal | cpu | torch.float32 | 0.68 | compiles_slower |
| ShapeSpec | wrapper | normal | cpu | torch.float32 | 0.67 | compiles_slower |
| Pad | wrapper | fwd | cuda | torch.complex64 | 0.67 | compiles_slower |
| ToeplitzPSF | wrapper | adj | cpu | torch.complex64 | 0.67 | compiles_slower |
| FastSamplingNUFFT | inner | adj | cuda | torch.complex64 | 0.67 | works_with_breaks |
| ShapeSpec | wrapper | fwd | cpu | torch.float32 | 0.66 | compiles_slower |
| ToeplitzPSF | wrapper | adj | cuda | torch.complex64 | 0.66 | compiles_slower |
| SumReduce | wrapper | adj | cpu | torch.complex64 | 0.66 | compiles_slower |
| SumReduce | wrapper | fwd | cuda | torch.complex64 | 0.66 | compiles_slower |
| Convolution | inner | fwd | cuda | torch.complex64 | 0.66 | compiles_slower |
| ToeplitzPSF | wrapper | adj | cuda | torch.complex64 | 0.66 | compiles_slower |
| ToeplitzPSF | wrapper | normal | cuda | torch.complex64 | 0.65 | compiles_slower |
| SamplingNUFFT | inner | normal | cuda | torch.complex64 | 0.64 | works_with_breaks |
| FFT | inner | normal | cuda | torch.complex64 | 0.64 | compiles_slower |
| Pad | wrapper | fwd | cpu | torch.complex64 | 0.64 | compiles_slower |
| Repeat | wrapper | fwd | cuda | torch.complex64 | 0.64 | compiles_slower |
| Zero | wrapper | normal | cuda | torch.complex64 | 0.63 | compiles_slower |
| FastSamplingNUFFT | wrapper | adj | cuda | torch.complex64 | 0.63 | works_with_breaks |
| Repeat | wrapper | adj | cuda | torch.complex64 | 0.63 | compiles_slower |
| Convolution | inner | fwd | cuda | torch.float32 | 0.63 | compiles_slower |
| Diagonal | wrapper | fwd | cuda | torch.float32 | 0.63 | compiles_slower |
| SamplingNUFFT | wrapper | adj | cuda | torch.complex64 | 0.63 | works_with_breaks |
| Convolution | inner | adj | cuda | torch.float32 | 0.62 | compiles_slower |
| SumReduce | wrapper | adj | cuda | torch.complex64 | 0.62 | compiles_slower |
| ToeplitzPSF | wrapper | fwd | cuda | torch.complex64 | 0.62 | compiles_slower |
| Diagonal | wrapper | adj | cuda | torch.float32 | 0.62 | compiles_slower |
| Convolution | inner | adj | cuda | torch.complex64 | 0.62 | compiles_slower |
| Diagonal | wrapper | normal | cuda | torch.float32 | 0.61 | compiles_slower |
| Truncate | wrapper | fwd | cpu | torch.complex64 | 0.61 | compiles_slower |
| PadDim | wrapper | adj | cpu | torch.complex64 | 0.6 | compiles_slower |
| Scalar | wrapper | fwd | cuda | torch.float32 | 0.6 | compiles_slower |
| SamplingNUFFT | inner | adj | cuda | torch.complex64 | 0.6 | works_with_breaks |
| Scalar | wrapper | normal | cuda | torch.float32 | 0.59 | compiles_slower |
| Rearrange | wrapper | adj | cpu | torch.complex64 | 0.59 | compiles_slower |
| Scalar | wrapper | adj | cuda | torch.float32 | 0.59 | compiles_slower |
| PadDim | wrapper | fwd | cpu | torch.complex64 | 0.59 | compiles_slower |
| Rearrange | wrapper | fwd | cuda | torch.complex64 | 0.58 | compiles_slower |
| Rearrange | wrapper | fwd | cpu | torch.complex64 | 0.58 | compiles_slower |
| Truncate | wrapper | adj | cpu | torch.complex64 | 0.58 | compiles_slower |
| Pad | wrapper | fwd | cuda | torch.complex64 | 0.57 | compiles_slower |
| Pad | wrapper | adj | cpu | torch.complex64 | 0.56 | compiles_slower |
| SamplingNUFFT | wrapper | normal | cuda | torch.complex64 | 0.56 | works_with_breaks |
| Zero | wrapper | normal | cuda | torch.complex64 | 0.55 | compiles_slower |
| SumReduce | wrapper | adj | cuda | torch.complex64 | 0.55 | compiles_slower |
| Repeat | wrapper | fwd | cuda | torch.complex64 | 0.55 | compiles_slower |
| Truncate | wrapper | fwd | cuda | torch.complex64 | 0.54 | compiles_slower |
| Convolution | inner | fwd | cuda | torch.float32 | 0.54 | compiles_slower |
| PadDim | wrapper | fwd | cuda | torch.complex64 | 0.54 | compiles_slower |
| Zero | wrapper | adj | cpu | torch.complex64 | 0.53 | compiles_slower |
| Rearrange | wrapper | adj | cuda | torch.complex64 | 0.53 | compiles_slower |
| ToeplitzPSF | inner | adj | cuda | torch.complex64 | 0.53 | compiles_slower |
| Zero | wrapper | fwd | cpu | torch.complex64 | 0.53 | compiles_slower |
| Truncate | wrapper | adj | cuda | torch.complex64 | 0.53 | compiles_slower |
| PadDim | wrapper | adj | cpu | torch.complex64 | 0.53 | compiles_slower |
| ToeplitzPSF | inner | fwd | cpu | torch.complex64 | 0.52 | compiles_slower |
| ToeplitzPSF | inner | adj | cpu | torch.complex64 | 0.52 | compiles_slower |
| PadDim | wrapper | fwd | cpu | torch.complex64 | 0.52 | compiles_slower |
| PadDim | wrapper | adj | cuda | torch.complex64 | 0.51 | compiles_slower |
| Truncate | wrapper | adj | cpu | torch.complex64 | 0.51 | compiles_slower |
| ToeplitzPSF | inner | fwd | cuda | torch.complex64 | 0.51 | compiles_slower |
| Truncate | wrapper | fwd | cpu | torch.complex64 | 0.5 | compiles_slower |
| Rearrange | wrapper | fwd | cuda | torch.complex64 | 0.5 | compiles_slower |
| Pad | wrapper | adj | cuda | torch.complex64 | 0.49 | compiles_slower |
| ToeplitzPSF | inner | adj | cpu | torch.complex64 | 0.48 | compiles_slower |
| Zero | wrapper | adj | cpu | torch.complex64 | 0.48 | compiles_slower |
| Rearrange | wrapper | adj | cuda | torch.complex64 | 0.48 | compiles_slower |
| PadDim | wrapper | fwd | cuda | torch.complex64 | 0.48 | compiles_slower |
| Pad | wrapper | adj | cpu | torch.complex64 | 0.48 | compiles_slower |
| ToeplitzPSF | inner | fwd | cpu | torch.complex64 | 0.47 | compiles_slower |
| ToeplitzPSF | inner | adj | cuda | torch.complex64 | 0.47 | compiles_slower |
| Truncate | wrapper | adj | cuda | torch.complex64 | 0.47 | compiles_slower |
| Zero | wrapper | fwd | cuda | torch.complex64 | 0.47 | compiles_slower |
| Zero | wrapper | fwd | cpu | torch.complex64 | 0.47 | compiles_slower |
| Truncate | wrapper | fwd | cuda | torch.complex64 | 0.47 | compiles_slower |
| Zero | wrapper | adj | cuda | torch.complex64 | 0.46 | compiles_slower |
| ToeplitzPSF | inner | fwd | cuda | torch.complex64 | 0.46 | compiles_slower |
| PadDim | wrapper | adj | cuda | torch.complex64 | 0.44 | compiles_slower |
| FFT | wrapper | fwd | cpu | torch.complex64 | 0.44 | compiles_slower |
| FFT | wrapper | adj | cuda | torch.complex64 | 0.44 | compiles_slower |
| FFT | wrapper | adj | cpu | torch.complex64 | 0.44 | compiles_slower |
| Pad | wrapper | adj | cuda | torch.complex64 | 0.43 | compiles_slower |
| FFT | wrapper | fwd | cpu | torch.complex64 | 0.42 | compiles_slower |
| Zero | wrapper | adj | cuda | torch.complex64 | 0.42 | compiles_slower |
| FFT | wrapper | adj | cuda | torch.complex64 | 0.42 | compiles_slower |
| FFT | wrapper | fwd | cuda | torch.complex64 | 0.41 | compiles_slower |
| FFT | wrapper | adj | cpu | torch.complex64 | 0.41 | compiles_slower |
| Zero | wrapper | fwd | cuda | torch.complex64 | 0.41 | compiles_slower |
| FFT | inner | adj | cpu | torch.complex64 | 0.4 | compiles_slower |
| FFT | wrapper | fwd | cuda | torch.complex64 | 0.4 | compiles_slower |
| FFT | inner | fwd | cpu | torch.complex64 | 0.4 | compiles_slower |
| FFT | inner | adj | cuda | torch.complex64 | 0.39 | compiles_slower |
| FFT | inner | fwd | cuda | torch.complex64 | 0.39 | compiles_slower |
| FFT | inner | fwd | cuda | torch.complex64 | 0.39 | compiles_slower |
| FFT | inner | fwd | cpu | torch.complex64 | 0.38 | compiles_slower |
| FFT | inner | adj | cuda | torch.complex64 | 0.38 | compiles_slower |
| FFT | inner | adj | cpu | torch.complex64 | 0.37 | compiles_slower |
| FFT | inner | normal | cuda | torch.complex64 | 0.25 | compiles_slower |
| FFT | inner | normal | cpu | torch.complex64 | 0.088 | compiles_slower |
| FFT | inner | normal | cpu | torch.complex64 | 0.018 | compiles_slower |

## Recompilation checks

| op | level | device | recompiles | agrees_after | max_rel_diff | shapes | notes |
|---|---|---|---|---|---|---|---|
| Dense | wrapper | cuda | - | - | - | [] | no batch-shaped input to vary |
| DenseSimple | wrapper | cuda | - | - | - | [] | no batch-shaped input to vary |
| Diagonal | wrapper | cuda | - | - | - | [] | eager rejected shape family: RuntimeError: The size of tensor a (20) must match the size of tensor b (10) at non-singlet |
| Identity | wrapper | cuda | 0 | True | 0 | [[4, 5], [8, 5]] | eager accepted both shapes |
| Zero | wrapper | cuda | 1 | True | 0 | [[4, 5], [8, 5]] | eager accepted both shapes |
| ShapeSpec | wrapper | cuda | 0 | True | 0 | [[4, 5], [8, 5]] | eager accepted both shapes |
| Scalar | wrapper | cuda | - | - | - | [] | no batch-shaped input to vary |
| Sleep | wrapper | cuda | - | - | - | [] | no batch-shaped input to vary |
| FFT | wrapper | cuda | 1 | True | 0 | [[8, 10], [16, 10]] | eager accepted both shapes |
| FFT | inner | cuda | 1 | True | 0 | [[8, 10], [16, 10]] | eager accepted both shapes |
| Convolution | wrapper | cuda | 1 | True | 0 | [[3, 16], [6, 16]] | eager accepted both shapes |
| Convolution | inner | cuda | 1 | True | 0 | [[3, 16], [6, 16]] | eager accepted both shapes |
| Interpolate | wrapper | cuda | - | - | - | [] | compiled shape change failed: InductorError: AssertionError: triton_kernel_wrapper_functional was not removed

Set TORCH |
| Interpolate | inner | cuda | - | - | - | [] | compiled shape change failed: InductorError: AssertionError: triton_kernel_wrapper_functional was not removed

Set TORCH |
| NUFFT | wrapper | cuda | - | - | - | [] | eager rejected shape family: RuntimeError: The size of tensor a (32) must match the size of tensor b (16) at non-singlet |
| NUFFT | inner | cuda | - | - | - | [] | eager rejected shape family: RuntimeError: The size of tensor a (32) must match the size of tensor b (16) at non-singlet |
| FastSamplingNUFFT | wrapper | cuda | - | - | - | [] | eager rejected shape family: ValueError: Mismatched shapes: expected (16, 16) but got torch.Size([32, 16]) |
| FastSamplingNUFFT | inner | cuda | - | - | - | [] | eager rejected shape family: ValueError: Mismatched shapes: expected (16, 16) but got torch.Size([32, 16]) |
| SamplingNUFFT | wrapper | cuda | - | - | - | [] | eager rejected shape family: ValueError: Mismatched shapes: expected (16, 16) but got torch.Size([32, 16]) |
| SamplingNUFFT | inner | cuda | - | - | - | [] | eager rejected shape family: ValueError: Mismatched shapes: expected (16, 16) but got torch.Size([32, 16]) |
| ToeplitzPSF | wrapper | cuda | 1 | True | 0 | [[2, 20, 20], [4, 20, 20]] | eager accepted both shapes |
| ToeplitzPSF | inner | cuda | 1 | True | 0 | [[2, 20, 20], [4, 20, 20]] | eager accepted both shapes |
| Sampling | wrapper | cuda | 1 | True | 0 | [[2, 16, 16], [4, 16, 16]] | eager accepted both shapes |
| ArrayToBlocks | wrapper | cuda | - | - | - | [] | compiled shape change failed: InductorError: SubprocException: An exception occurred in a subprocess:

Name=_unfold2d
Tr |
| BlocksToArray | wrapper | cuda | - | - | - | [] | eager rejected shape family: RuntimeError: Fold expected input with full size (2, 2, 4, 8) but got torch.Size([1, 4, 2,  |
| Pad | wrapper | cuda | - | - | - | [] | eager rejected shape family: ValueError: Mismatched shapes: expected (10, 10) but got torch.Size([20, 10]) |
| Truncate | wrapper | cuda | 1 | True | 0 | [[4, 5], [8, 5]] | eager accepted both shapes |
| PadDim | wrapper | cuda | 1 | True | 0 | [[4, 3], [8, 3]] | eager accepted both shapes |
| Rearrange | wrapper | cuda | - | - | - | [] | eager rejected shape family: EinopsError:  Error while processing rearrange-reduction pattern "(A B) C -> A B C".
 Input |
| SumReduce | wrapper | cuda | 1 | True | 0 | [[5, 2, 3], [10, 2, 3]] | eager accepted both shapes |
| Repeat | wrapper | cuda | 1 | True | 0 | [[4, 5], [8, 5]] | eager accepted both shapes |
| Add | wrapper | cuda | - | - | - | [] | no batch-shaped input to vary |
| Stack | wrapper | cuda | - | - | - | [] | no batch-shaped input to vary |
| Concat | wrapper | cuda | - | - | - | [] | compiled shape change failed: InternalTorchDynamoError: NotImplementedError: currently can't reconstruct arbitrary froze |
| Chain | wrapper | cuda | - | - | - | [] | no batch-shaped input to vary |
| ToDevice | wrapper | cuda | - | - | - | [] | no batch-shaped input to vary |
| SimpleLinop | wrapper | cuda | - | - | - | [] | no batch-shaped input to vary |
| functional.unfold | inner | cuda | - | - | - | [] | compiled shape change failed: InductorError: SubprocException: An exception occurred in a subprocess:

Name=_unfold2d
Tr |
| functional.fold | inner | cuda | - | - | - | [] | compiled shape change failed: InductorError: SubprocException: An exception occurred in a subprocess:

Name=_fold2d
Trac |
| functional.grid | inner | cuda | - | - | - | [] | no batch-shaped input to vary |
| functional.ungrid | inner | cuda | - | - | - | [] | compiled shape change failed: InductorError: AssertionError: triton_kernel_wrapper_functional was not removed

Set TORCH |

## Autograd checks (single-op backward)

| op | level | class | max_rel_diff | notes |
|---|---|---|---|---|
| FFT | wrapper | works | 0 |  |
| FFT | inner | works | 0 |  |
| Convolution | wrapper | works | 0 |  |
| Convolution | inner | works | 0 |  |
| Interpolate | wrapper | fails_loud | - | raised during compiled backward |
| Interpolate | inner | fails_loud | - | raised during compiled backward |
| NUFFT | wrapper | fails_loud | - | raised during compiled backward |
| NUFFT | inner | fails_loud | - | raised during compiled backward |
| FastSamplingNUFFT | wrapper | works | 0 |  |
| FastSamplingNUFFT | inner | works | 0 |  |
| SamplingNUFFT | wrapper | works | 0 |  |
| SamplingNUFFT | inner | works | 0 |  |
| ToeplitzPSF | wrapper | works | 0 |  |
| ToeplitzPSF | inner | works | 0 |  |
| functional.unfold | inner | fails_loud | - | raised during eager backward |
| functional.fold | inner | fails_loud | - | raised during eager backward |
| functional.grid | inner | fails_loud | - | raised during eager backward |
| functional.ungrid | inner | fails_loud | - | raised during eager backward |

## Pipeline checks

| pipeline | level | config | class | speedup | breaks | notes |
|---|---|---|---|---|---|---|
| pipeline.simple | loop | eager | works | - | 0 | 20-step fixed point |
| pipeline.simple | loop | default | works | 0.99 | 2 | break-kind: wrapper (logging/namedlinop); recompiles covered by the dedicated phase |
| pipeline.simple | loop | fullgraph | fails_loud | - | 0 |  |
| pipeline.simple | compiled-inners | default | works | 1.1 | 0 | eager loop with individually compiled linops |
| pipeline.nufft | loop | eager | works | - | 0 | 20-step fixed point |
| pipeline.nufft | loop | default | fails_loud | - | 0 |  |
| pipeline.nufft | loop | fullgraph | fails_loud | - | 0 |  |
| pipeline.nufft | compiled-inners | default | fails_loud | - | 0 |  |

## Deep-dive observations (auto-collected)

- **FastSamplingNUFFT** (inner, fullgraph):
  - x4: `Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"`
- **FastSamplingNUFFT** (wrapper, fullgraph):
  - x5: `Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"`
- **Interpolate** (inner, default):
  - x3: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **Interpolate** (inner, fullgraph):
  - x3: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **Interpolate** (wrapper, default):
  - x3: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **Interpolate** (wrapper, fullgraph):
  - x3: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **NUFFT** (inner, default):
  - x3: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **NUFFT** (inner, fullgraph):
  - x3: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **NUFFT** (wrapper, default):
  - x3: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **NUFFT** (wrapper, fullgraph):
  - x3: `Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"`
- **SamplingNUFFT** (inner, fullgraph):
  - x4: `Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"`
- **SamplingNUFFT** (wrapper, fullgraph):
  - x5: `Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"`
- **functional.fold** (inner, default):
  - x1: `Set TORCHDYNAMO_VERBOSE=1 for the full Dynamo stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"`
- **functional.fold** (inner, fullgraph):
  - x1: `Set TORCHDYNAMO_VERBOSE=1 for the full Dynamo stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"`
- **functional.grid** (inner, default):
  - x1: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **functional.grid** (inner, fullgraph):
  - x1: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **functional.unfold** (inner, default):
  - x1: `Set TORCHDYNAMO_VERBOSE=1 for the full Dynamo stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"`
- **functional.unfold** (inner, fullgraph):
  - x1: `Set TORCHDYNAMO_VERBOSE=1 for the full Dynamo stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"`
- **functional.ungrid** (inner, default):
  - x1: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`
- **functional.ungrid** (inner, fullgraph):
  - x1: `torch._inductor.exc.InductorError: AssertionError: triton_kernel_wrapper_functional was not removed`

Mechanism notes (hand-written): TBD

## Implicated root causes (hand-written)

TBD

