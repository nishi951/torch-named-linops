# CPU Benchmarks

Performance benchmarks on CPU.

## Metadata

- **Date**: 2026-07-07T12:28:06.113308
- **Commit**: `26b2481`
- **Working tree**: clean
- **PyTorch**: 2.10.0+cu128
- **CUDA**: 12.8
- **GPU**: NVIDIA GeForce RTX 3090
- **Python**: 3.10.16
- **OS**: Linux
- **Threads**: 12

## ArrayToBlocks 3D

| Operation | Size | Library | Mean | Median | IQR | Peak Memory |
|-----------|------|---------|------|--------|-----|-------------|
| ArrayToBlocks forward | 32x32x32 | torchlinops | 4.79 ms | 4.05 ms | 30.53 us | — |
| ArrayToBlocks forward | 32x32x32 | torchlinops (linop) | 4.06 ms | 4.06 ms | 32.66 us | — |
| ArrayToBlocks forward | 32x32x32 | sigpy | 173.69 us | 172.56 us | 2.15 us | — |
| ArrayToBlocks forward | 64x64x64 | torchlinops | 36.50 ms | 36.47 ms | 154.50 us | — |
| ArrayToBlocks forward | 64x64x64 | torchlinops (linop) | 36.52 ms | 36.50 ms | 98.44 us | — |
| ArrayToBlocks forward | 64x64x64 | sigpy | 2.62 ms | 2.62 ms | 69.78 us | — |
| ArrayToBlocks forward | 128x128x128 | torchlinops | 324.68 ms | 324.93 ms | 1.41 ms | — |
| ArrayToBlocks forward | 128x128x128 | torchlinops (linop) | 325.34 ms | 325.45 ms | 1.62 ms | — |
| ArrayToBlocks forward | 128x128x128 | sigpy | 34.55 ms | 34.44 ms | 353.71 us | — |
| BlocksToArray forward | 32x32x32 | torchlinops | 5.38 ms | 5.37 ms | 26.49 us | — |
| BlocksToArray forward | 32x32x32 | torchlinops (linop) | 5.37 ms | 5.38 ms | 32.44 us | — |
| BlocksToArray forward | 32x32x32 | sigpy | 1.37 ms | 1.37 ms | 11.97 us | — |
| BlocksToArray forward | 64x64x64 | torchlinops | 49.45 ms | 49.49 ms | 118.50 us | — |
| BlocksToArray forward | 64x64x64 | torchlinops (linop) | 49.57 ms | 49.51 ms | 301.88 us | — |
| BlocksToArray forward | 64x64x64 | sigpy | 12.11 ms | 12.10 ms | 427.77 us | — |
| BlocksToArray forward | 128x128x128 | torchlinops | 424.69 ms | 424.75 ms | 1.30 ms | — |
| BlocksToArray forward | 128x128x128 | torchlinops (linop) | 425.22 ms | 425.18 ms | 1.56 ms | — |
| BlocksToArray forward | 128x128x128 | sigpy | 124.77 ms | 124.86 ms | 2.73 ms | — |


## Interpolate 2D

| Operation | Size | Library | Mean | Median | IQR | Peak Memory |
|-----------|------|---------|------|--------|-----|-------------|
| Interpolate forward 2D | 64x64 | torchlinops | 1.68 ms | 1.68 ms | 14.97 us | — |
| Interpolate forward 2D | 64x64 | torchlinops (linop) | 1.69 ms | 1.68 ms | 6.90 us | — |
| Interpolate forward 2D | 64x64 | sigpy | 726.62 us | 724.30 us | 2.75 us | — |
| Interpolate forward 2D | 128x128 | torchlinops | 4.14 ms | 4.14 ms | 10.34 us | — |
| Interpolate forward 2D | 128x128 | torchlinops (linop) | 4.17 ms | 4.17 ms | 26.39 us | — |
| Interpolate forward 2D | 128x128 | sigpy | 3.00 ms | 2.94 ms | 67.90 us | — |
| Interpolate forward 2D | 256x256 | torchlinops | 19.98 ms | 19.97 ms | 33.34 us | — |
| Interpolate forward 2D | 256x256 | torchlinops (linop) | 19.98 ms | 19.98 ms | 69.88 us | — |
| Interpolate forward 2D | 256x256 | sigpy | 11.47 ms | 11.46 ms | 17.55 us | — |
| Interpolate adjoint 2D | 64x64 | torchlinops | 1.90 ms | 1.90 ms | 13.79 us | — |
| Interpolate adjoint 2D | 64x64 | torchlinops (linop) | 1.92 ms | 1.92 ms | 9.96 us | — |
| Interpolate adjoint 2D | 64x64 | sigpy | 734.75 us | 732.01 us | 2.05 us | — |
| Interpolate adjoint 2D | 128x128 | torchlinops | 5.14 ms | 5.14 ms | 17.90 us | — |
| Interpolate adjoint 2D | 128x128 | torchlinops (linop) | 5.19 ms | 5.17 ms | 23.31 us | — |
| Interpolate adjoint 2D | 128x128 | sigpy | 2.92 ms | 2.92 ms | 24.43 us | — |
| Interpolate adjoint 2D | 256x256 | torchlinops | 23.41 ms | 23.42 ms | 47.01 us | — |
| Interpolate adjoint 2D | 256x256 | torchlinops (linop) | 23.39 ms | 23.41 ms | 40.84 us | — |
| Interpolate adjoint 2D | 256x256 | sigpy | 11.60 ms | 11.57 ms | 48.51 us | — |


## Interpolate 3D

| Operation | Size | Library | Mean | Median | IQR | Peak Memory |
|-----------|------|---------|------|--------|-----|-------------|
| Interpolate forward 3D | 32x32x32 | torchlinops | 7.64 ms | 7.63 ms | 65.02 us | — |
| Interpolate forward 3D | 32x32x32 | torchlinops (linop) | 7.63 ms | 7.62 ms | 26.58 us | — |
| Interpolate forward 3D | 32x32x32 | sigpy | 3.34 ms | 3.33 ms | 16.31 us | — |
| Interpolate forward 3D | 64x64x64 | torchlinops | 88.41 ms | 88.72 ms | 3.89 ms | — |
| Interpolate forward 3D | 64x64x64 | torchlinops (linop) | 89.15 ms | 89.23 ms | 1.88 ms | — |
| Interpolate forward 3D | 64x64x64 | sigpy | 26.93 ms | 26.86 ms | 766.18 us | — |
| Interpolate forward 3D | 128x128x128 | torchlinops | 645.20 ms | 643.79 ms | 2.78 ms | — |
| Interpolate forward 3D | 128x128x128 | torchlinops (linop) | 651.00 ms | 653.58 ms | 22.08 ms | — |
| Interpolate forward 3D | 128x128x128 | sigpy | 264.24 ms | 263.83 ms | 2.55 ms | — |
| Interpolate adjoint 3D | 32x32x32 | torchlinops | 8.94 ms | 8.94 ms | 74.79 us | — |
| Interpolate adjoint 3D | 32x32x32 | torchlinops (linop) | 8.96 ms | 8.96 ms | 27.06 us | — |
| Interpolate adjoint 3D | 32x32x32 | sigpy | 3.35 ms | 3.36 ms | 26.48 us | — |
| Interpolate adjoint 3D | 64x64x64 | torchlinops | 100.72 ms | 100.68 ms | 4.78 ms | — |
| Interpolate adjoint 3D | 64x64x64 | torchlinops (linop) | 99.15 ms | 99.55 ms | 2.89 ms | — |
| Interpolate adjoint 3D | 64x64x64 | sigpy | 26.11 ms | 26.09 ms | 28.40 us | — |
| Interpolate adjoint 3D | 128x128x128 | torchlinops | 739.47 ms | 735.59 ms | 30.79 ms | — |
| Interpolate adjoint 3D | 128x128x128 | torchlinops (linop) | 741.82 ms | 744.53 ms | 11.70 ms | — |
| Interpolate adjoint 3D | 128x128x128 | sigpy | 260.37 ms | 260.05 ms | 2.40 ms | — |


## NUFFT 2D

| Operation | Size | Library | Mean | Median | IQR | Peak Memory |
|-----------|------|---------|------|--------|-----|-------------|
| NUFFT forward 2D | 64x64 | torchlinops | 3.56 ms | 3.55 ms | 20.24 us | — |
| NUFFT forward 2D | 64x64 | torchlinops (linop) | 3.11 ms | 3.10 ms | 52.96 us | — |
| NUFFT forward 2D | 64x64 | sigpy | 2.30 ms | 2.30 ms | 10.64 us | — |
| NUFFT forward 2D | 128x128 | torchlinops | 6.28 ms | 6.27 ms | 47.36 us | — |
| NUFFT forward 2D | 128x128 | torchlinops (linop) | 5.94 ms | 5.93 ms | 27.47 us | — |
| NUFFT forward 2D | 128x128 | sigpy | 8.84 ms | 8.83 ms | 29.64 us | — |
| NUFFT forward 2D | 256x256 | torchlinops | 28.43 ms | 28.42 ms | 141.59 us | — |
| NUFFT forward 2D | 256x256 | torchlinops (linop) | 25.97 ms | 25.94 ms | 80.28 us | — |
| NUFFT forward 2D | 256x256 | sigpy | 35.04 ms | 34.95 ms | 183.87 us | — |
| NUFFT adjoint 2D | 64x64 | torchlinops | 2.26 ms | 2.25 ms | 15.74 us | — |
| NUFFT adjoint 2D | 64x64 | torchlinops (linop) | 2.13 ms | 2.13 ms | 13.33 us | — |
| NUFFT adjoint 2D | 64x64 | sigpy | 2.36 ms | 2.36 ms | 16.45 us | — |
| NUFFT adjoint 2D | 128x128 | torchlinops | 5.83 ms | 5.83 ms | 47.39 us | — |
| NUFFT adjoint 2D | 128x128 | torchlinops (linop) | 5.41 ms | 5.41 ms | 13.71 us | — |
| NUFFT adjoint 2D | 128x128 | sigpy | 9.14 ms | 9.16 ms | 120.87 us | — |
| NUFFT adjoint 2D | 256x256 | torchlinops | 28.81 ms | 28.83 ms | 103.24 us | — |
| NUFFT adjoint 2D | 256x256 | torchlinops (linop) | 27.88 ms | 27.86 ms | 185.05 us | — |
| NUFFT adjoint 2D | 256x256 | sigpy | 36.15 ms | 35.98 ms | 432.57 us | — |


## NUFFT 3D

| Operation | Size | Library | Mean | Median | IQR | Peak Memory |
|-----------|------|---------|------|--------|-----|-------------|
| NUFFT forward 3D | 32x32x32 | torchlinops | 8.61 ms | 8.62 ms | 34.16 us | — |
| NUFFT forward 3D | 32x32x32 | torchlinops (linop) | 8.09 ms | 8.08 ms | 34.41 us | — |
| NUFFT forward 3D | 32x32x32 | sigpy | 9.91 ms | 9.90 ms | 17.42 us | — |
| NUFFT forward 3D | 64x64x64 | torchlinops | 99.45 ms | 102.38 ms | 8.42 ms | — |
| NUFFT forward 3D | 64x64x64 | torchlinops (linop) | 92.24 ms | 91.65 ms | 2.21 ms | — |
| NUFFT forward 3D | 64x64x64 | sigpy | 80.29 ms | 80.15 ms | 375.71 us | — |
| NUFFT forward 3D | 128x128x128 | torchlinops | 715.23 ms | 714.61 ms | 9.87 ms | — |
| NUFFT forward 3D | 128x128x128 | torchlinops (linop) | 678.05 ms | 680.67 ms | 8.48 ms | — |
| NUFFT forward 3D | 128x128x128 | sigpy | 770.44 ms | 770.07 ms | 2.76 ms | — |
| NUFFT adjoint 3D | 32x32x32 | torchlinops | 9.80 ms | 9.79 ms | 41.61 us | — |
| NUFFT adjoint 3D | 32x32x32 | torchlinops (linop) | 9.53 ms | 9.53 ms | 11.05 us | — |
| NUFFT adjoint 3D | 32x32x32 | sigpy | 10.19 ms | 10.18 ms | 30.88 us | — |
| NUFFT adjoint 3D | 64x64x64 | torchlinops | 104.49 ms | 104.48 ms | 2.17 ms | — |
| NUFFT adjoint 3D | 64x64x64 | torchlinops (linop) | 102.07 ms | 104.10 ms | 4.63 ms | — |
| NUFFT adjoint 3D | 64x64x64 | sigpy | 82.29 ms | 82.19 ms | 564.61 us | — |
| NUFFT adjoint 3D | 128x128x128 | torchlinops | 900.42 ms | 900.15 ms | 13.58 ms | — |
| NUFFT adjoint 3D | 128x128x128 | torchlinops (linop) | 870.37 ms | 869.46 ms | 7.85 ms | — |
| NUFFT adjoint 3D | 128x128x128 | sigpy | 837.04 ms | 836.74 ms | 1.22 ms | — |


## Bar Charts

### Small

![Timing bar chart (small)](assets/timing_cpu_small.png)

### Medium

![Timing bar chart (medium)](assets/timing_cpu_medium.png)

### Large

![Timing bar chart (large)](assets/timing_cpu_large.png)

## Scaling Curves

![Timing scaling](assets/scaling_time_cpu.png)
