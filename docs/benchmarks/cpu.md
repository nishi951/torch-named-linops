# CPU Benchmarks

Performance benchmarks on CPU.

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
| ArrayToBlocks forward | 32x32x32 | torchlinops | 4.08 ms | 129.54 us | 4.08 ms | 36.58 us | — |
| ArrayToBlocks forward | 32x32x32 | torchlinops (linop) | 4.10 ms | 129.67 us | 4.10 ms | 20.54 us | — |
| ArrayToBlocks forward | 32x32x32 | sigpy | 175.95 us | 734.63 us | 174.54 us | 1.89 us | — |
| ArrayToBlocks forward | 64x64x64 | torchlinops | 37.12 ms | 1.02 ms | 37.08 ms | 328.55 us | — |
| ArrayToBlocks forward | 64x64x64 | torchlinops (linop) | 37.24 ms | 1.02 ms | 37.30 ms | 271.57 us | — |
| ArrayToBlocks forward | 64x64x64 | sigpy | 2.84 ms | 5.61 ms | 2.77 ms | 220.30 us | — |
| ArrayToBlocks forward | 128x128x128 | torchlinops | 330.90 ms | 8.19 ms | 330.76 ms | 518.05 us | — |
| ArrayToBlocks forward | 128x128x128 | torchlinops (linop) | 328.89 ms | 8.20 ms | 329.16 ms | 1.55 ms | — |
| ArrayToBlocks forward | 128x128x128 | sigpy | 35.09 ms | 48.58 ms | 34.93 ms | 165.34 us | — |
| BlocksToArray forward | 32x32x32 | torchlinops | 5.48 ms | 693.53 us | 5.50 ms | 71.31 us | — |
| BlocksToArray forward | 32x32x32 | torchlinops (linop) | 5.49 ms | 683.51 us | 5.48 ms | 68.20 us | — |
| BlocksToArray forward | 32x32x32 | sigpy | 1.36 ms | 3.83 ms | 1.36 ms | 4.73 us | — |
| BlocksToArray forward | 64x64x64 | torchlinops | 49.47 ms | 6.81 ms | 49.49 ms | 513.58 us | — |
| BlocksToArray forward | 64x64x64 | torchlinops (linop) | 49.40 ms | 6.74 ms | 49.39 ms | 197.97 us | — |
| BlocksToArray forward | 64x64x64 | sigpy | 12.28 ms | 38.37 ms | 12.26 ms | 137.31 us | — |
| BlocksToArray forward | 128x128x128 | torchlinops | 425.83 ms | 86.31 ms | 426.36 ms | 1.14 ms | — |
| BlocksToArray forward | 128x128x128 | torchlinops (linop) | 423.58 ms | 86.44 ms | 423.53 ms | 2.07 ms | — |
| BlocksToArray forward | 128x128x128 | sigpy | 131.94 ms | 381.13 ms | 132.22 ms | 1.93 ms | — |


## Interpolate 2D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| Interpolate forward 2D | 64x64 | torchlinops | 1.69 ms | 17.31 us | 1.69 ms | 13.18 us | — |
| Interpolate forward 2D | 64x64 | torchlinops (linop) | 1.70 ms | 17.29 us | 1.70 ms | 6.57 us | — |
| Interpolate forward 2D | 64x64 | sigpy | 724.38 us | 92.44 us | 723.11 us | 2.00 us | — |
| Interpolate forward 2D | 128x128 | torchlinops | 4.33 ms | 64.94 us | 4.32 ms | 29.35 us | — |
| Interpolate forward 2D | 128x128 | torchlinops (linop) | 4.31 ms | 64.82 us | 4.31 ms | 13.24 us | — |
| Interpolate forward 2D | 128x128 | sigpy | 2.90 ms | 361.87 us | 2.89 ms | 18.86 us | — |
| Interpolate forward 2D | 256x256 | torchlinops | 24.54 ms | 254.76 us | 24.55 ms | 84.24 us | — |
| Interpolate forward 2D | 256x256 | torchlinops (linop) | 24.59 ms | 255.06 us | 24.58 ms | 79.60 us | — |
| Interpolate forward 2D | 256x256 | sigpy | 11.52 ms | 1.48 ms | 11.49 ms | 47.14 us | — |
| Interpolate adjoint 2D | 64x64 | torchlinops | 1.89 ms | 17.28 us | 1.89 ms | 15.18 us | — |
| Interpolate adjoint 2D | 64x64 | torchlinops (linop) | 1.92 ms | 17.28 us | 1.92 ms | 13.60 us | — |
| Interpolate adjoint 2D | 64x64 | sigpy | 731.20 us | 90.20 us | 729.75 us | 1.68 us | — |
| Interpolate adjoint 2D | 128x128 | torchlinops | 5.20 ms | 64.73 us | 5.19 ms | 19.84 us | — |
| Interpolate adjoint 2D | 128x128 | torchlinops (linop) | 5.23 ms | 65.85 us | 5.22 ms | 12.78 us | — |
| Interpolate adjoint 2D | 128x128 | sigpy | 2.95 ms | 365.70 us | 2.94 ms | 53.35 us | — |
| Interpolate adjoint 2D | 256x256 | torchlinops | 27.87 ms | 254.57 us | 27.86 ms | 81.83 us | — |
| Interpolate adjoint 2D | 256x256 | torchlinops (linop) | 27.96 ms | 255.07 us | 27.96 ms | 94.24 us | — |
| Interpolate adjoint 2D | 256x256 | sigpy | 11.58 ms | 1.49 ms | 11.56 ms | 12.74 us | — |


## Interpolate 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| Interpolate forward 3D | 32x32x32 | torchlinops | 7.66 ms | 129.04 us | 7.66 ms | 12.79 us | — |
| Interpolate forward 3D | 32x32x32 | torchlinops (linop) | 7.69 ms | 128.17 us | 7.68 ms | 20.28 us | — |
| Interpolate forward 3D | 32x32x32 | sigpy | 3.32 ms | 699.78 us | 3.32 ms | 8.00 us | — |
| Interpolate forward 3D | 64x64x64 | torchlinops | 93.22 ms | 1.02 ms | 92.88 ms | 1.83 ms | — |
| Interpolate forward 3D | 64x64x64 | torchlinops (linop) | 94.03 ms | 1.03 ms | 94.27 ms | 2.16 ms | — |
| Interpolate forward 3D | 64x64x64 | sigpy | 26.52 ms | 5.97 ms | 26.48 ms | 65.27 us | — |
| Interpolate forward 3D | 128x128x128 | torchlinops | 664.20 ms | 8.23 ms | 668.55 ms | 22.82 ms | — |
| Interpolate forward 3D | 128x128x128 | torchlinops (linop) | 659.19 ms | 8.23 ms | 659.87 ms | 23.95 ms | — |
| Interpolate forward 3D | 128x128x128 | sigpy | 262.52 ms | 48.85 ms | 262.82 ms | 2.36 ms | — |
| Interpolate adjoint 3D | 32x32x32 | torchlinops | 11.51 ms | 17.25 us | 11.50 ms | 40.38 us | — |
| Interpolate adjoint 3D | 32x32x32 | torchlinops (linop) | 10.47 ms | 17.29 us | 10.06 ms | 1.22 ms | — |
| Interpolate adjoint 3D | 32x32x32 | sigpy | 3.33 ms | 96.57 us | 3.32 ms | 50.91 us | — |
| Interpolate adjoint 3D | 64x64x64 | torchlinops | 106.07 ms | 128.05 us | 106.13 ms | 5.26 ms | — |
| Interpolate adjoint 3D | 64x64x64 | torchlinops (linop) | 103.25 ms | 127.99 us | 103.12 ms | 4.09 ms | — |
| Interpolate adjoint 3D | 64x64x64 | sigpy | 26.23 ms | 698.00 us | 26.15 ms | 106.49 us | — |
| Interpolate adjoint 3D | 128x128x128 | torchlinops | 759.94 ms | 1.02 ms | 757.83 ms | 10.70 ms | — |
| Interpolate adjoint 3D | 128x128x128 | torchlinops (linop) | 753.98 ms | 1.02 ms | 756.15 ms | 17.93 ms | — |
| Interpolate adjoint 3D | 128x128x128 | sigpy | 272.19 ms | 5.95 ms | 271.46 ms | 2.02 ms | — |


## NUFFT 2D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| NUFFT forward 2D | 64x64 | torchlinops | 2.58 ms | 17.32 us | 2.58 ms | 29.30 us | — |
| NUFFT forward 2D | 64x64 | torchlinops (linop) | 2.44 ms | 17.43 us | 2.41 ms | 62.93 us | — |
| NUFFT forward 2D | 64x64 | sigpy | 2.26 ms | 89.51 us | 2.25 ms | 18.86 us | — |
| NUFFT forward 2D | 128x128 | torchlinops | 6.32 ms | 64.77 us | 6.32 ms | 15.74 us | — |
| NUFFT forward 2D | 128x128 | torchlinops (linop) | 6.03 ms | 65.03 us | 6.04 ms | 40.27 us | — |
| NUFFT forward 2D | 128x128 | sigpy | 8.94 ms | 355.19 us | 8.90 ms | 124.80 us | — |
| NUFFT forward 2D | 256x256 | torchlinops | 27.00 ms | 256.31 us | 27.01 ms | 241.22 us | — |
| NUFFT forward 2D | 256x256 | torchlinops (linop) | 24.97 ms | 254.67 us | 25.00 ms | 128.93 us | — |
| NUFFT forward 2D | 256x256 | sigpy | 35.00 ms | 1.46 ms | 34.92 ms | 171.73 us | — |
| NUFFT adjoint 2D | 64x64 | torchlinops | 2.24 ms | 17.35 us | 2.24 ms | 19.46 us | — |
| NUFFT adjoint 2D | 64x64 | torchlinops (linop) | 2.12 ms | 17.42 us | 2.11 ms | 23.89 us | — |
| NUFFT adjoint 2D | 64x64 | sigpy | 2.36 ms | 92.59 us | 2.36 ms | 7.35 us | — |
| NUFFT adjoint 2D | 128x128 | torchlinops | 5.86 ms | 64.71 us | 5.85 ms | 40.84 us | — |
| NUFFT adjoint 2D | 128x128 | torchlinops (linop) | 5.38 ms | 65.04 us | 5.37 ms | 34.74 us | — |
| NUFFT adjoint 2D | 128x128 | sigpy | 9.13 ms | 360.58 us | 9.09 ms | 99.54 us | — |
| NUFFT adjoint 2D | 256x256 | torchlinops | 30.68 ms | 254.75 us | 30.67 ms | 237.72 us | — |
| NUFFT adjoint 2D | 256x256 | torchlinops (linop) | 28.52 ms | 254.37 us | 28.58 ms | 235.54 us | — |
| NUFFT adjoint 2D | 256x256 | sigpy | 36.07 ms | 1.47 ms | 35.94 ms | 218.04 us | — |


## NUFFT 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| NUFFT forward 3D | 32x32x32 | torchlinops | 8.71 ms | 128.79 us | 8.66 ms | 122.89 us | — |
| NUFFT forward 3D | 32x32x32 | torchlinops (linop) | 8.13 ms | 128.00 us | 8.12 ms | 51.05 us | — |
| NUFFT forward 3D | 32x32x32 | sigpy | 9.94 ms | 714.85 us | 9.93 ms | 43.18 us | — |
| NUFFT forward 3D | 64x64x64 | torchlinops | 97.26 ms | 1.02 ms | 97.71 ms | 2.46 ms | — |
| NUFFT forward 3D | 64x64x64 | torchlinops (linop) | 97.37 ms | 1.03 ms | 98.14 ms | 6.69 ms | — |
| NUFFT forward 3D | 64x64x64 | sigpy | 80.32 ms | 6.09 ms | 80.16 ms | 786.44 us | — |
| NUFFT forward 3D | 128x128x128 | torchlinops | 721.78 ms | 8.29 ms | 723.68 ms | 18.90 ms | — |
| NUFFT forward 3D | 128x128x128 | torchlinops (linop) | 686.62 ms | 8.28 ms | 687.24 ms | 11.20 ms | — |
| NUFFT forward 3D | 128x128x128 | sigpy | 768.84 ms | 48.74 ms | 768.49 ms | 2.34 ms | — |
| NUFFT adjoint 3D | 32x32x32 | torchlinops | 10.15 ms | 17.32 us | 10.09 ms | 106.66 us | — |
| NUFFT adjoint 3D | 32x32x32 | torchlinops (linop) | 9.68 ms | 17.26 us | 9.67 ms | 44.35 us | — |
| NUFFT adjoint 3D | 32x32x32 | sigpy | 10.21 ms | 93.44 us | 10.19 ms | 33.37 us | — |
| NUFFT adjoint 3D | 64x64x64 | torchlinops | 109.20 ms | 128.56 us | 109.19 ms | 1.29 ms | — |
| NUFFT adjoint 3D | 64x64x64 | torchlinops (linop) | 106.00 ms | 128.18 us | 106.41 ms | 1.93 ms | — |
| NUFFT adjoint 3D | 64x64x64 | sigpy | 82.96 ms | 702.76 us | 82.74 ms | 322.33 us | — |
| NUFFT adjoint 3D | 128x128x128 | torchlinops | 925.57 ms | 1.02 ms | 927.11 ms | 8.30 ms | — |
| NUFFT adjoint 3D | 128x128x128 | torchlinops (linop) | 883.67 ms | 1.02 ms | 881.36 ms | 6.57 ms | — |
| NUFFT adjoint 3D | 128x128x128 | sigpy | 836.92 ms | 5.78 ms | 836.68 ms | 784.60 us | — |


## Bar Charts

### Small

![Timing bar chart (small)](assets/timing_cpu_small.png)

### Medium

![Timing bar chart (medium)](assets/timing_cpu_medium.png)

### Large

![Timing bar chart (large)](assets/timing_cpu_large.png)

## Scaling Curves

![Timing scaling](assets/scaling_time_cpu.png)
