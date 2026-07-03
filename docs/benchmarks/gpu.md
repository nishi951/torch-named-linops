# GPU Benchmarks

Performance benchmarks on GPU.

## Metadata

- **Date**: 2026-07-03T14:56:46.736391
- **Commit**: `4ee7153`
- **Working tree**: dirty ([patch.diff](../../benchmarks/results/latest/patch.diff))
- **PyTorch**: 2.10.0+cu128
- **CUDA**: 12.8
- **GPU**: NVIDIA GeForce RTX 3090
- **Python**: 3.10.16
- **OS**: Linux
- **Threads**: 12

## ArrayToBlocks 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| ArrayToBlocks forward | 32x32x32 | torchlinops | 113.25 us | 2.46 us | 112.64 us | 2.43 us | 1.59 MB |
| ArrayToBlocks forward | 32x32x32 | torchlinops (linop) | 124.06 us | 2.46 us | 123.23 us | 2.78 us | 1.59 MB |
| ArrayToBlocks forward | 32x32x32 | sigpy | 70.69 us | 28.45 us | 70.52 us | 628.00 ns | 3.45 MB |
| ArrayToBlocks forward | 64x64x64 | torchlinops | 113.18 us | 2.99 us | 112.64 us | 2.05 us | 16.00 MB |
| ArrayToBlocks forward | 64x64x64 | torchlinops (linop) | 124.95 us | 2.76 us | 124.10 us | 3.07 us | 16.00 MB |
| ArrayToBlocks forward | 64x64x64 | sigpy | 113.10 us | 28.33 us | 113.93 us | 166.88 ns | 19.70 MB |
| ArrayToBlocks forward | 128x128x128 | torchlinops | 386.59 us | 24.96 us | 386.05 us | 1.82 us | 132.37 MB |
| ArrayToBlocks forward | 128x128x128 | torchlinops (linop) | 396.37 us | 24.98 us | 396.29 us | 2.05 us | 132.37 MB |
| ArrayToBlocks forward | 128x128x128 | sigpy | 934.17 us | 163.53 us | 921.47 us | 25.69 us | 164.89 MB |
| BlocksToArray forward | 32x32x32 | torchlinops | 98.95 us | 2.92 us | 98.30 us | 1.86 us | 2.68 MB |
| BlocksToArray forward | 32x32x32 | torchlinops (linop) | 111.10 us | 2.93 us | 110.59 us | 2.21 us | 2.68 MB |
| BlocksToArray forward | 32x32x32 | sigpy | 69.18 us | 28.87 us | 68.46 us | 1.69 us | 4.54 MB |
| BlocksToArray forward | 64x64x64 | torchlinops | 98.24 us | 20.01 us | 98.30 us | 2.24 us | 26.37 MB |
| BlocksToArray forward | 64x64x64 | torchlinops (linop) | 109.25 us | 19.85 us | 108.54 us | 2.21 us | 26.37 MB |
| BlocksToArray forward | 64x64x64 | sigpy | 162.06 us | 130.58 us | 161.76 us | 558.00 ns | 40.07 MB |
| BlocksToArray forward | 128x128x128 | torchlinops | 321.37 us | 185.08 us | 322.19 us | 1.02 us | 232.74 MB |
| BlocksToArray forward | 128x128x128 | torchlinops (linop) | 303.45 us | 163.91 us | 301.12 us | 9.22 us | 232.74 MB |
| BlocksToArray forward | 128x128x128 | sigpy | 1.38 ms | 1.14 ms | 1.28 ms | 6.98 us | 349.63 MB |


## Interpolate 2D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| Interpolate forward 2D | 64x64 | torchlinops | 97.78 us | 24.59 us | 97.28 us | 2.14 us | 192.00 KB |
| Interpolate forward 2D | 64x64 | torchlinops (linop) | 104.11 us | 2.44 us | 103.42 us | 2.14 us | 96.00 KB |
| Interpolate forward 2D | 64x64 | sigpy | 189.35 us | 89.80 us | 189.25 us | 1.38 us | 528.00 KB |
| Interpolate forward 2D | 128x128 | torchlinops | 127.62 us | 24.49 us | 126.98 us | 1.22 us | 768.00 KB |
| Interpolate forward 2D | 128x128 | torchlinops (linop) | 134.51 us | 2.43 us | 134.14 us | 2.05 us | 384.00 KB |
| Interpolate forward 2D | 128x128 | sigpy | 189.25 us | 87.75 us | 188.42 us | 647.99 ns | 528.00 KB |
| Interpolate forward 2D | 256x256 | torchlinops | 275.27 us | 24.56 us | 275.46 us | 2.14 us | 3.00 MB |
| Interpolate forward 2D | 256x256 | torchlinops (linop) | 284.12 us | 2.85 us | 283.65 us | 1.86 us | 1.50 MB |
| Interpolate forward 2D | 256x256 | sigpy | 262.00 us | 87.53 us | 261.63 us | 1.09 us | 1.77 MB |
| Interpolate adjoint 2D | 64x64 | torchlinops | 96.57 us | 24.72 us | 96.26 us | 2.05 us | 192.00 KB |
| Interpolate adjoint 2D | 64x64 | torchlinops (linop) | 103.83 us | 2.41 us | 103.42 us | 1.98 us | 96.00 KB |
| Interpolate adjoint 2D | 64x64 | sigpy | 186.58 us | 84.91 us | 186.53 us | 800.01 ns | 528.00 KB |
| Interpolate adjoint 2D | 128x128 | torchlinops | 123.63 us | 24.50 us | 122.88 us | 1.15 us | 768.00 KB |
| Interpolate adjoint 2D | 128x128 | torchlinops (linop) | 131.07 us | 2.47 us | 130.94 us | 2.08 us | 384.00 KB |
| Interpolate adjoint 2D | 128x128 | sigpy | 186.55 us | 83.91 us | 186.33 us | 1.03 us | 528.00 KB |
| Interpolate adjoint 2D | 256x256 | torchlinops | 279.67 us | 24.45 us | 279.42 us | 2.02 us | 3.00 MB |
| Interpolate adjoint 2D | 256x256 | torchlinops (linop) | 265.60 us | 2.85 us | 265.22 us | 2.05 us | 1.50 MB |
| Interpolate adjoint 2D | 256x256 | sigpy | 262.57 us | 83.51 us | 262.16 us | 1.13 us | 1.77 MB |


## Interpolate 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| Interpolate forward 3D | 32x32x32 | torchlinops | 110.74 us | 25.24 us | 110.53 us | 1.89 us | 704.00 KB |
| Interpolate forward 3D | 32x32x32 | torchlinops (linop) | 116.55 us | 2.46 us | 115.71 us | 1.73 us | 560.00 KB |
| Interpolate forward 3D | 32x32x32 | sigpy | 203.52 us | 98.71 us | 202.98 us | 2.10 us | 1.02 MB |
| Interpolate forward 3D | 64x64x64 | torchlinops | 344.14 us | 25.19 us | 344.06 us | 2.05 us | 5.50 MB |
| Interpolate forward 3D | 64x64x64 | torchlinops (linop) | 357.91 us | 2.77 us | 357.38 us | 5.41 us | 4.38 MB |
| Interpolate forward 3D | 64x64x64 | sigpy | 570.98 us | 97.62 us | 570.61 us | 1.95 us | 6.52 MB |
| Interpolate forward 3D | 128x128x128 | torchlinops | 2.41 ms | 85.77 us | 2.42 ms | 67.58 us | 46.00 MB |
| Interpolate forward 3D | 128x128x128 | torchlinops (linop) | 2.61 ms | 24.91 us | 2.64 ms | 70.66 us | 35.00 MB |
| Interpolate forward 3D | 128x128x128 | sigpy | 3.37 ms | 205.58 us | 3.34 ms | 19.60 us | 48.52 MB |
| Interpolate adjoint 3D | 32x32x32 | torchlinops | 110.40 us | 24.95 us | 109.82 us | 1.15 us | 384.00 KB |
| Interpolate adjoint 3D | 32x32x32 | torchlinops (linop) | 117.51 us | 2.46 us | 116.96 us | 1.12 us | 336.00 KB |
| Interpolate adjoint 3D | 32x32x32 | sigpy | 203.62 us | 97.18 us | 202.86 us | 1.71 us | 528.00 KB |
| Interpolate adjoint 3D | 64x64x64 | torchlinops | 329.28 us | 25.16 us | 331.09 us | 8.42 us | 3.00 MB |
| Interpolate adjoint 3D | 64x64x64 | torchlinops (linop) | 339.69 us | 2.48 us | 342.91 us | 8.19 us | 2.62 MB |
| Interpolate adjoint 3D | 64x64x64 | sigpy | 559.44 us | 96.03 us | 558.05 us | 2.42 us | 4.77 MB |
| Interpolate adjoint 3D | 128x128x128 | torchlinops | 2.31 ms | 22.41 us | 2.32 ms | 17.63 us | 26.00 MB |
| Interpolate adjoint 3D | 128x128x128 | torchlinops (linop) | 2.29 ms | 2.79 us | 2.29 ms | 10.43 us | 21.00 MB |
| Interpolate adjoint 3D | 128x128x128 | sigpy | 3.30 ms | 95.34 us | 3.28 ms | 7.79 us | 22.52 MB |


## NUFFT 2D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| NUFFT forward 2D | 64x64 | torchlinops | 453.28 us | 27.64 us | 451.58 us | 12.29 us | 392.50 KB |
| NUFFT forward 2D | 64x64 | torchlinops (linop) | 284.60 us | 2.43 us | 283.65 us | 9.38 us | 312.50 KB |
| NUFFT forward 2D | 64x64 | sigpy | 648.70 us | 115.18 us | 647.51 us | 3.23 us | 262.00 KB |
| NUFFT forward 2D | 128x128 | torchlinops | 578.22 us | 27.51 us | 576.51 us | 11.10 us | 1.53 MB |
| NUFFT forward 2D | 128x128 | torchlinops (linop) | 288.10 us | 2.41 us | 287.49 us | 8.26 us | 1.22 MB |
| NUFFT forward 2D | 128x128 | sigpy | 718.97 us | 115.10 us | 718.34 us | 2.40 us | 1000.00 KB |
| NUFFT forward 2D | 256x256 | torchlinops | 1.06 ms | 28.19 us | 1.04 ms | 13.09 us | 6.13 MB |
| NUFFT forward 2D | 256x256 | torchlinops (linop) | 434.99 us | 2.80 us | 435.02 us | 9.31 us | 4.88 MB |
| NUFFT forward 2D | 256x256 | sigpy | 1.18 ms | 116.20 us | 1.17 ms | 4.67 us | 3.86 MB |
| NUFFT adjoint 2D | 64x64 | torchlinops | 446.23 us | 27.56 us | 445.25 us | 11.46 us | 392.50 KB |
| NUFFT adjoint 2D | 64x64 | torchlinops (linop) | 271.03 us | 2.42 us | 269.78 us | 9.89 us | 312.50 KB |
| NUFFT adjoint 2D | 64x64 | sigpy | 648.17 us | 111.50 us | 648.01 us | 4.49 us | 262.00 KB |
| NUFFT adjoint 2D | 128x128 | torchlinops | 563.42 us | 27.43 us | 562.18 us | 10.53 us | 1.53 MB |
| NUFFT adjoint 2D | 128x128 | torchlinops (linop) | 268.04 us | 2.43 us | 267.26 us | 9.22 us | 1.22 MB |
| NUFFT adjoint 2D | 128x128 | sigpy | 659.68 us | 112.65 us | 652.42 us | 12.87 us | 1000.00 KB |
| NUFFT adjoint 2D | 256x256 | torchlinops | 987.53 us | 28.18 us | 985.28 us | 13.25 us | 6.13 MB |
| NUFFT adjoint 2D | 256x256 | torchlinops (linop) | 364.64 us | 2.92 us | 363.52 us | 7.10 us | 4.88 MB |
| NUFFT adjoint 2D | 256x256 | sigpy | 934.24 us | 112.99 us | 933.76 us | 3.10 us | 3.86 MB |


## NUFFT 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| NUFFT forward 3D | 32x32x32 | torchlinops | 878.75 us | 23.17 us | 874.21 us | 20.77 us | 3.13 MB |
| NUFFT forward 3D | 32x32x32 | torchlinops (linop) | 294.74 us | 2.44 us | 293.89 us | 10.02 us | 2.91 MB |
| NUFFT forward 3D | 32x32x32 | sigpy | 1.08 ms | 129.19 us | 1.08 ms | 7.42 us | 2.23 MB |
| NUFFT forward 3D | 64x64x64 | torchlinops | 2.02 ms | 23.28 us | 2.02 ms | 16.29 us | 25.41 MB |
| NUFFT forward 3D | 64x64x64 | torchlinops (linop) | 520.08 us | 2.81 us | 519.17 us | 13.22 us | 23.66 MB |
| NUFFT forward 3D | 64x64x64 | sigpy | 2.20 ms | 128.27 us | 2.19 ms | 10.69 us | 18.23 MB |
| NUFFT forward 3D | 128x128x128 | torchlinops | 31.51 ms | 34.86 us | 31.51 ms | 98.11 us | 204.00 MB |
| NUFFT forward 3D | 128x128x128 | torchlinops (linop) | 3.60 ms | 20.59 us | 3.59 ms | 41.98 us | 190.00 MB |
| NUFFT forward 3D | 128x128x128 | sigpy | 10.71 ms | 246.29 us | 10.63 ms | 10.02 us | 142.27 MB |
| NUFFT adjoint 3D | 32x32x32 | torchlinops | 661.75 us | 22.87 us | 659.47 us | 12.29 us | 2.91 MB |
| NUFFT adjoint 3D | 32x32x32 | torchlinops (linop) | 280.38 us | 2.48 us | 279.55 us | 10.24 us | 2.69 MB |
| NUFFT adjoint 3D | 32x32x32 | sigpy | 806.82 us | 126.01 us | 806.01 us | 4.57 us | 1.98 MB |
| NUFFT adjoint 3D | 64x64x64 | torchlinops | 2.10 ms | 23.12 us | 2.08 ms | 18.46 us | 23.66 MB |
| NUFFT adjoint 3D | 64x64x64 | torchlinops (linop) | 495.33 us | 2.47 us | 495.62 us | 6.02 us | 21.91 MB |
| NUFFT adjoint 3D | 64x64x64 | sigpy | 1.89 ms | 125.97 us | 1.88 ms | 19.01 us | 12.86 MB |
| NUFFT adjoint 3D | 128x128x128 | torchlinops | 32.34 ms | 28.40 us | 32.33 ms | 173.06 us | 190.00 MB |
| NUFFT adjoint 3D | 128x128x128 | torchlinops (linop) | 3.41 ms | 2.80 us | 3.41 ms | 7.17 us | 176.00 MB |
| NUFFT adjoint 3D | 128x128x128 | sigpy | 10.60 ms | 132.67 us | 10.51 ms | 25.51 us | 103.27 MB |


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
