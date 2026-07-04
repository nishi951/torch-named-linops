# GPU Benchmarks

Performance benchmarks on GPU.

## Metadata

- **Date**: 2026-07-04T01:58:24.995183
- **Commit**: `16b8203`
- **Working tree**: clean
- **PyTorch**: 2.10.0+cu128
- **CUDA**: 12.8
- **GPU**: NVIDIA GeForce RTX 3090
- **Python**: 3.10.16
- **OS**: Linux
- **Threads**: 12

## ArrayToBlocks 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| ArrayToBlocks forward | 32x32x32 | torchlinops | 121.38 us | 7.82 us | 120.83 us | 2.85 us | 1.59 MB |
| ArrayToBlocks forward | 32x32x32 | torchlinops (linop) | 134.45 us | 7.90 us | 133.12 us | 4.10 us | 1.59 MB |
| ArrayToBlocks forward | 32x32x32 | sigpy | 48.42 us | 34.35 us | 48.13 us | 1.28 us | 2.48 MB |
| ArrayToBlocks forward | 64x64x64 | torchlinops | 120.99 us | 7.83 us | 120.77 us | 2.19 us | 16.00 MB |
| ArrayToBlocks forward | 64x64x64 | torchlinops (linop) | 133.27 us | 7.86 us | 132.90 us | 2.91 us | 16.00 MB |
| ArrayToBlocks forward | 64x64x64 | sigpy | 112.11 us | 33.10 us | 111.62 us | 1.02 us | 21.70 MB |
| ArrayToBlocks forward | 128x128x128 | torchlinops | 392.55 us | 25.05 us | 392.19 us | 2.05 us | 132.37 MB |
| ArrayToBlocks forward | 128x128x128 | torchlinops (linop) | 402.68 us | 24.86 us | 402.43 us | 2.05 us | 132.37 MB |
| ArrayToBlocks forward | 128x128x128 | sigpy | 725.81 us | 158.49 us | 721.92 us | 1.02 us | 180.89 MB |
| BlocksToArray forward | 32x32x32 | torchlinops | 107.53 us | 7.93 us | 106.59 us | 3.03 us | 2.68 MB |
| BlocksToArray forward | 32x32x32 | torchlinops (linop) | 121.00 us | 8.01 us | 118.85 us | 3.28 us | 2.68 MB |
| BlocksToArray forward | 32x32x32 | sigpy | 44.27 us | 34.37 us | 44.03 us | 1.50 us | 5.88 MB |
| BlocksToArray forward | 64x64x64 | torchlinops | 106.49 us | 20.30 us | 106.18 us | 2.06 us | 26.37 MB |
| BlocksToArray forward | 64x64x64 | torchlinops (linop) | 116.63 us | 20.11 us | 115.94 us | 3.07 us | 26.37 MB |
| BlocksToArray forward | 64x64x64 | sigpy | 32.41 us | 130.27 us | 32.77 us | 1.02 us | 53.25 MB |
| BlocksToArray forward | 128x128x128 | torchlinops | 321.19 us | 200.06 us | 321.54 us | 904.00 ns | 232.74 MB |
| BlocksToArray forward | 128x128x128 | torchlinops (linop) | 301.63 us | 160.86 us | 301.06 us | 8.19 us | 232.74 MB |
| BlocksToArray forward | 128x128x128 | sigpy | 220.31 us | 1.11 ms | 216.06 us | 10.27 us | 466.00 MB |


## Interpolate 2D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| Interpolate forward 2D | 64x64 | torchlinops | 99.26 us | 7.85 us | 98.34 us | 2.78 us | 96.00 KB |
| Interpolate forward 2D | 64x64 | torchlinops (linop) | 113.35 us | 7.80 us | 112.64 us | 2.35 us | 96.00 KB |
| Interpolate forward 2D | 64x64 | sigpy | 111.93 us | 48.74 us | 111.62 us | 2.05 us | 528.00 KB |
| Interpolate forward 2D | 128x128 | torchlinops | 135.71 us | 7.82 us | 135.17 us | 2.05 us | 384.00 KB |
| Interpolate forward 2D | 128x128 | torchlinops (linop) | 148.35 us | 7.86 us | 147.46 us | 3.02 us | 384.00 KB |
| Interpolate forward 2D | 128x128 | sigpy | 145.16 us | 47.41 us | 144.38 us | 2.05 us | 656.00 KB |
| Interpolate forward 2D | 256x256 | torchlinops | 295.86 us | 7.80 us | 294.91 us | 1.34 us | 1.50 MB |
| Interpolate forward 2D | 256x256 | torchlinops (linop) | 291.46 us | 7.85 us | 291.34 us | 4.05 us | 1.50 MB |
| Interpolate forward 2D | 256x256 | sigpy | 292.13 us | 45.87 us | 291.78 us | 1.98 us | 2.27 MB |
| Interpolate adjoint 2D | 64x64 | torchlinops | 99.54 us | 7.85 us | 99.14 us | 2.05 us | 96.00 KB |
| Interpolate adjoint 2D | 64x64 | torchlinops (linop) | 112.82 us | 7.82 us | 111.84 us | 2.80 us | 96.00 KB |
| Interpolate adjoint 2D | 64x64 | sigpy | 112.61 us | 45.24 us | 111.90 us | 2.27 us | 528.00 KB |
| Interpolate adjoint 2D | 128x128 | torchlinops | 130.88 us | 7.83 us | 130.05 us | 1.22 us | 384.00 KB |
| Interpolate adjoint 2D | 128x128 | torchlinops (linop) | 142.21 us | 7.82 us | 142.02 us | 2.88 us | 384.00 KB |
| Interpolate adjoint 2D | 128x128 | sigpy | 146.26 us | 44.28 us | 145.47 us | 2.73 us | 656.00 KB |
| Interpolate adjoint 2D | 256x256 | torchlinops | 280.39 us | 7.83 us | 280.51 us | 2.05 us | 1.50 MB |
| Interpolate adjoint 2D | 256x256 | torchlinops (linop) | 271.85 us | 7.82 us | 271.36 us | 2.82 us | 1.50 MB |
| Interpolate adjoint 2D | 256x256 | sigpy | 292.26 us | 43.85 us | 291.81 us | 1.92 us | 2.27 MB |


## Interpolate 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| Interpolate forward 3D | 32x32x32 | torchlinops | 110.25 us | 7.87 us | 109.57 us | 1.38 us | 560.00 KB |
| Interpolate forward 3D | 32x32x32 | torchlinops (linop) | 123.51 us | 7.92 us | 122.88 us | 2.05 us | 560.00 KB |
| Interpolate forward 3D | 32x32x32 | sigpy | 214.64 us | 47.03 us | 214.02 us | 2.75 us | 1.14 MB |
| Interpolate forward 3D | 64x64x64 | torchlinops | 367.23 us | 7.87 us | 366.59 us | 2.05 us | 4.38 MB |
| Interpolate forward 3D | 64x64x64 | torchlinops (linop) | 361.54 us | 7.90 us | 360.13 us | 2.72 us | 4.38 MB |
| Interpolate forward 3D | 64x64x64 | sigpy | 576.35 us | 46.53 us | 575.49 us | 2.93 us | 8.52 MB |
| Interpolate forward 3D | 128x128x128 | torchlinops | 2.64 ms | 30.78 us | 2.58 ms | 176.64 us | 35.00 MB |
| Interpolate forward 3D | 128x128x128 | torchlinops (linop) | 2.62 ms | 24.76 us | 2.62 ms | 18.43 us | 35.00 MB |
| Interpolate forward 3D | 128x128x128 | sigpy | 3.18 ms | 160.15 us | 3.16 ms | 49.92 us | 64.52 MB |
| Interpolate adjoint 3D | 32x32x32 | torchlinops | 110.00 us | 7.83 us | 109.57 us | 2.05 us | 336.00 KB |
| Interpolate adjoint 3D | 32x32x32 | torchlinops (linop) | 122.97 us | 7.94 us | 122.56 us | 2.18 us | 336.00 KB |
| Interpolate adjoint 3D | 32x32x32 | sigpy | 211.64 us | 43.55 us | 210.94 us | 1.34 us | 528.00 KB |
| Interpolate adjoint 3D | 64x64x64 | torchlinops | 354.96 us | 7.86 us | 354.30 us | 2.05 us | 2.62 MB |
| Interpolate adjoint 3D | 64x64x64 | torchlinops (linop) | 344.80 us | 7.83 us | 346.11 us | 8.72 us | 2.62 MB |
| Interpolate adjoint 3D | 64x64x64 | sigpy | 578.04 us | 44.48 us | 576.66 us | 3.07 us | 3.14 MB |
| Interpolate adjoint 3D | 128x128x128 | torchlinops | 2.32 ms | 7.88 us | 2.31 ms | 9.27 us | 21.00 MB |
| Interpolate adjoint 3D | 128x128x128 | torchlinops (linop) | 2.28 ms | 7.84 us | 2.27 ms | 11.64 us | 21.00 MB |
| Interpolate adjoint 3D | 128x128x128 | sigpy | 3.19 ms | 42.28 us | 3.18 ms | 12.73 us | 24.52 MB |


## NUFFT 2D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| NUFFT forward 2D | 64x64 | torchlinops | 549.75 us | 8.77 us | 548.86 us | 12.58 us | 360.50 KB |
| NUFFT forward 2D | 64x64 | torchlinops (linop) | 361.94 us | 8.79 us | 357.38 us | 13.41 us | 312.50 KB |
| NUFFT forward 2D | 64x64 | sigpy | 562.99 us | 44.89 us | 561.25 us | 11.50 us | 294.00 KB |
| NUFFT forward 2D | 128x128 | torchlinops | 652.85 us | 7.76 us | 651.26 us | 11.10 us | 1.41 MB |
| NUFFT forward 2D | 128x128 | torchlinops (linop) | 344.09 us | 7.79 us | 343.04 us | 9.68 us | 1.22 MB |
| NUFFT forward 2D | 128x128 | sigpy | 661.73 us | 45.12 us | 660.88 us | 12.46 us | 1.10 MB |
| NUFFT forward 2D | 256x256 | torchlinops | 1.10 ms | 7.78 us | 1.10 ms | 15.62 us | 5.63 MB |
| NUFFT forward 2D | 256x256 | torchlinops (linop) | 480.25 us | 7.78 us | 477.98 us | 11.26 us | 4.88 MB |
| NUFFT forward 2D | 256x256 | sigpy | 1.09 ms | 46.14 us | 1.07 ms | 19.62 us | 4.36 MB |
| NUFFT adjoint 2D | 64x64 | torchlinops | 507.58 us | 7.75 us | 505.94 us | 13.38 us | 360.50 KB |
| NUFFT adjoint 2D | 64x64 | torchlinops (linop) | 328.84 us | 7.68 us | 326.72 us | 12.19 us | 312.50 KB |
| NUFFT adjoint 2D | 64x64 | sigpy | 487.35 us | 43.32 us | 484.35 us | 13.65 us | 294.00 KB |
| NUFFT adjoint 2D | 128x128 | torchlinops | 630.66 us | 7.77 us | 628.74 us | 14.18 us | 1.41 MB |
| NUFFT adjoint 2D | 128x128 | torchlinops (linop) | 325.95 us | 7.72 us | 325.63 us | 11.18 us | 1.22 MB |
| NUFFT adjoint 2D | 128x128 | sigpy | 479.14 us | 43.37 us | 478.40 us | 12.26 us | 1.10 MB |
| NUFFT adjoint 2D | 256x256 | torchlinops | 1.02 ms | 7.81 us | 1.01 ms | 21.44 us | 5.63 MB |
| NUFFT adjoint 2D | 256x256 | torchlinops (linop) | 372.83 us | 7.76 us | 371.71 us | 6.40 us | 4.88 MB |
| NUFFT adjoint 2D | 256x256 | sigpy | 844.82 us | 43.15 us | 844.80 us | 5.12 us | 4.36 MB |


## NUFFT 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| NUFFT forward 3D | 32x32x32 | torchlinops | 963.51 us | 7.82 us | 957.44 us | 17.74 us | 3.08 MB |
| NUFFT forward 3D | 32x32x32 | torchlinops (linop) | 361.33 us | 7.88 us | 360.13 us | 12.29 us | 2.91 MB |
| NUFFT forward 3D | 32x32x32 | sigpy | 993.00 us | 46.69 us | 980.99 us | 6.42 us | 2.12 MB |
| NUFFT forward 3D | 64x64x64 | torchlinops | 2.15 ms | 7.83 us | 2.15 ms | 14.76 us | 25.03 MB |
| NUFFT forward 3D | 64x64x64 | torchlinops (linop) | 565.94 us | 7.84 us | 565.06 us | 9.22 us | 23.66 MB |
| NUFFT forward 3D | 64x64x64 | sigpy | 2.08 ms | 46.20 us | 2.06 ms | 11.39 us | 20.23 MB |
| NUFFT forward 3D | 128x128x128 | torchlinops | 34.11 ms | 24.68 us | 34.18 ms | 182.75 us | 202.00 MB |
| NUFFT forward 3D | 128x128x128 | torchlinops (linop) | 3.51 ms | 24.49 us | 3.52 ms | 14.27 us | 190.00 MB |
| NUFFT forward 3D | 128x128x128 | sigpy | 10.30 ms | 157.75 us | 10.25 ms | 112.64 us | 158.27 MB |
| NUFFT adjoint 3D | 32x32x32 | torchlinops | 756.98 us | 7.80 us | 747.39 us | 15.36 us | 2.86 MB |
| NUFFT adjoint 3D | 32x32x32 | torchlinops (linop) | 349.41 us | 7.70 us | 349.18 us | 9.22 us | 2.69 MB |
| NUFFT adjoint 3D | 32x32x32 | sigpy | 655.03 us | 44.59 us | 654.34 us | 6.14 us | 1.98 MB |
| NUFFT adjoint 3D | 64x64x64 | torchlinops | 2.07 ms | 7.80 us | 2.06 ms | 12.54 us | 23.28 MB |
| NUFFT adjoint 3D | 64x64x64 | torchlinops (linop) | 509.05 us | 7.77 us | 507.90 us | 7.97 us | 21.91 MB |
| NUFFT adjoint 3D | 64x64x64 | sigpy | 1.80 ms | 43.86 us | 1.79 ms | 8.34 us | 12.86 MB |
| NUFFT adjoint 3D | 128x128x128 | torchlinops | 33.79 ms | 7.77 us | 33.95 ms | 519.17 us | 187.00 MB |
| NUFFT adjoint 3D | 128x128x128 | torchlinops (linop) | 3.46 ms | 7.70 us | 3.46 ms | 9.73 us | 176.00 MB |
| NUFFT adjoint 3D | 128x128x128 | sigpy | 10.41 ms | 44.18 us | 10.40 ms | 11.56 us | 105.27 MB |


## Bar Charts

### Small

![Timing bar chart (small)](assets/timing_cuda_small.png)

### Medium

![Timing bar chart (medium)](assets/timing_cuda_medium.png)

### Large

![Timing bar chart (large)](assets/timing_cuda_large.png)

## Scaling Curves

![Timing scaling](assets/scaling_time_cuda.png)

![Memory scaling](assets/scaling_memory.png)
