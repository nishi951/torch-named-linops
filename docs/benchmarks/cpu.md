# CPU Benchmarks

Performance benchmarks on CPU.

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
| ArrayToBlocks forward | 32x32x32 | torchlinops | 6.20 ms | 25.80 us | 4.07 ms | 7.03 ms | — |
| ArrayToBlocks forward | 32x32x32 | torchlinops (linop) | 4.07 ms | 130.01 us | 4.06 ms | 23.51 us | — |
| ArrayToBlocks forward | 32x32x32 | sigpy | 173.70 us | 291.97 us | 172.53 us | 2.77 us | — |
| ArrayToBlocks forward | 64x64x64 | torchlinops | 36.51 ms | 211.67 us | 36.49 ms | 406.54 us | — |
| ArrayToBlocks forward | 64x64x64 | torchlinops (linop) | 36.78 ms | 211.60 us | 36.73 ms | 360.30 us | — |
| ArrayToBlocks forward | 64x64x64 | sigpy | 2.62 ms | 2.61 ms | 2.62 ms | 64.66 us | — |
| ArrayToBlocks forward | 128x128x128 | torchlinops | 325.35 ms | 3.78 ms | 324.99 ms | 1.30 ms | — |
| ArrayToBlocks forward | 128x128x128 | torchlinops (linop) | 324.53 ms | 3.78 ms | 324.62 ms | 1.38 ms | — |
| ArrayToBlocks forward | 128x128x128 | sigpy | 34.88 ms | 48.54 ms | 34.80 ms | 201.97 us | — |
| BlocksToArray forward | 32x32x32 | torchlinops | 5.33 ms | 279.25 us | 5.34 ms | 50.94 us | — |
| BlocksToArray forward | 32x32x32 | torchlinops (linop) | 5.40 ms | 277.89 us | 5.41 ms | 34.63 us | — |
| BlocksToArray forward | 32x32x32 | sigpy | 1.36 ms | 2.53 ms | 1.36 ms | 4.90 us | — |
| BlocksToArray forward | 64x64x64 | torchlinops | 48.95 ms | 3.11 ms | 48.98 ms | 221.33 us | — |
| BlocksToArray forward | 64x64x64 | torchlinops (linop) | 49.06 ms | 3.10 ms | 49.05 ms | 403.87 us | — |
| BlocksToArray forward | 64x64x64 | sigpy | 12.39 ms | 38.44 ms | 12.40 ms | 311.87 us | — |
| BlocksToArray forward | 128x128x128 | torchlinops | 424.00 ms | 86.40 ms | 424.89 ms | 3.10 ms | — |
| BlocksToArray forward | 128x128x128 | torchlinops (linop) | 424.48 ms | 86.32 ms | 424.48 ms | 1.72 ms | — |
| BlocksToArray forward | 128x128x128 | sigpy | 132.12 ms | 383.52 ms | 132.38 ms | 3.02 ms | — |


## Interpolate 2D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| Interpolate forward 2D | 64x64 | torchlinops | 1.69 ms | 22.94 us | 1.69 ms | 13.66 us | — |
| Interpolate forward 2D | 64x64 | torchlinops (linop) | 1.71 ms | 17.30 us | 1.71 ms | 10.12 us | — |
| Interpolate forward 2D | 64x64 | sigpy | 728.02 us | 133.70 us | 726.14 us | 2.53 us | — |
| Interpolate forward 2D | 128x128 | torchlinops | 4.21 ms | 154.62 us | 4.20 ms | 14.63 us | — |
| Interpolate forward 2D | 128x128 | torchlinops (linop) | 4.25 ms | 25.96 us | 4.24 ms | 49.82 us | — |
| Interpolate forward 2D | 128x128 | sigpy | 2.88 ms | 271.59 us | 2.88 ms | 15.00 us | — |
| Interpolate forward 2D | 256x256 | torchlinops | 24.01 ms | 242.17 us | 23.94 ms | 221.24 us | — |
| Interpolate forward 2D | 256x256 | torchlinops (linop) | 24.45 ms | 205.21 us | 24.46 ms | 99.93 us | — |
| Interpolate forward 2D | 256x256 | sigpy | 12.15 ms | 1.89 ms | 12.15 ms | 7.28 us | — |
| Interpolate adjoint 2D | 64x64 | torchlinops | 1.92 ms | 23.06 us | 1.91 ms | 15.20 us | — |
| Interpolate adjoint 2D | 64x64 | torchlinops (linop) | 1.93 ms | 17.23 us | 1.93 ms | 14.76 us | — |
| Interpolate adjoint 2D | 64x64 | sigpy | 734.23 us | 26.49 us | 733.60 us | 2.73 us | — |
| Interpolate adjoint 2D | 128x128 | torchlinops | 5.20 ms | 153.28 us | 5.20 ms | 29.69 us | — |
| Interpolate adjoint 2D | 128x128 | torchlinops (linop) | 5.21 ms | 25.99 us | 5.20 ms | 48.68 us | — |
| Interpolate adjoint 2D | 128x128 | sigpy | 2.91 ms | 270.49 us | 2.91 ms | 19.90 us | — |
| Interpolate adjoint 2D | 256x256 | torchlinops | 27.08 ms | 240.36 us | 27.05 ms | 316.91 us | — |
| Interpolate adjoint 2D | 256x256 | torchlinops (linop) | 27.21 ms | 206.73 us | 27.17 ms | 190.23 us | — |
| Interpolate adjoint 2D | 256x256 | sigpy | 12.24 ms | 1.86 ms | 12.24 ms | 16.36 us | — |


## Interpolate 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| Interpolate forward 3D | 32x32x32 | torchlinops | 7.53 ms | 185.61 us | 7.51 ms | 73.39 us | — |
| Interpolate forward 3D | 32x32x32 | torchlinops (linop) | 7.55 ms | 25.70 us | 7.55 ms | 40.51 us | — |
| Interpolate forward 3D | 32x32x32 | sigpy | 3.32 ms | 313.36 us | 3.32 ms | 11.64 us | — |
| Interpolate forward 3D | 64x64x64 | torchlinops | 90.77 ms | 264.90 us | 90.62 ms | 3.70 ms | — |
| Interpolate forward 3D | 64x64x64 | torchlinops (linop) | 92.42 ms | 212.98 us | 92.68 ms | 2.10 ms | — |
| Interpolate forward 3D | 64x64x64 | sigpy | 26.57 ms | 2.85 ms | 26.51 ms | 177.03 us | — |
| Interpolate forward 3D | 128x128x128 | torchlinops | 657.02 ms | 2.98 ms | 657.73 ms | 4.98 ms | — |
| Interpolate forward 3D | 128x128x128 | torchlinops (linop) | 651.51 ms | 3.82 ms | 646.55 ms | 24.63 ms | — |
| Interpolate forward 3D | 128x128x128 | sigpy | 264.13 ms | 51.81 ms | 263.68 ms | 1.74 ms | — |
| Interpolate adjoint 3D | 32x32x32 | torchlinops | 8.95 ms | 30.03 us | 8.94 ms | 57.84 us | — |
| Interpolate adjoint 3D | 32x32x32 | torchlinops (linop) | 9.04 ms | 17.35 us | 9.02 ms | 63.57 us | — |
| Interpolate adjoint 3D | 32x32x32 | sigpy | 3.28 ms | 149.38 us | 3.27 ms | 11.43 us | — |
| Interpolate adjoint 3D | 64x64x64 | torchlinops | 104.84 ms | 230.15 us | 104.83 ms | 6.91 ms | — |
| Interpolate adjoint 3D | 64x64x64 | torchlinops (linop) | 102.97 ms | 25.74 us | 102.92 ms | 4.93 ms | — |
| Interpolate adjoint 3D | 64x64x64 | sigpy | 26.22 ms | 401.88 us | 26.15 ms | 218.76 us | — |
| Interpolate adjoint 3D | 128x128x128 | torchlinops | 757.43 ms | 2.35 ms | 756.33 ms | 15.74 ms | — |
| Interpolate adjoint 3D | 128x128x128 | torchlinops (linop) | 748.21 ms | 211.88 us | 744.81 ms | 14.92 ms | — |
| Interpolate adjoint 3D | 128x128x128 | sigpy | 279.38 ms | 3.76 ms | 279.13 ms | 3.99 ms | — |


## NUFFT 2D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| NUFFT forward 2D | 64x64 | torchlinops | 2.59 ms | 69.41 us | 2.60 ms | 24.75 us | — |
| NUFFT forward 2D | 64x64 | torchlinops (linop) | 2.45 ms | 17.41 us | 2.45 ms | 30.01 us | — |
| NUFFT forward 2D | 64x64 | sigpy | 2.32 ms | 143.74 us | 2.32 ms | 16.36 us | — |
| NUFFT forward 2D | 128x128 | torchlinops | 6.42 ms | 178.70 us | 6.42 ms | 170.38 us | — |
| NUFFT forward 2D | 128x128 | torchlinops (linop) | 6.05 ms | 25.97 us | 6.04 ms | 47.05 us | — |
| NUFFT forward 2D | 128x128 | sigpy | 8.85 ms | 284.76 us | 8.84 ms | 26.94 us | — |
| NUFFT forward 2D | 256x256 | torchlinops | 26.47 ms | 272.37 us | 26.43 ms | 173.95 us | — |
| NUFFT forward 2D | 256x256 | torchlinops (linop) | 24.58 ms | 205.09 us | 24.54 ms | 159.22 us | — |
| NUFFT forward 2D | 256x256 | sigpy | 35.69 ms | 1.90 ms | 35.60 ms | 206.97 us | — |
| NUFFT adjoint 2D | 64x64 | torchlinops | 2.27 ms | 27.43 us | 2.26 ms | 23.93 us | — |
| NUFFT adjoint 2D | 64x64 | torchlinops (linop) | 2.12 ms | 17.34 us | 2.12 ms | 13.76 us | — |
| NUFFT adjoint 2D | 64x64 | sigpy | 2.38 ms | 147.30 us | 2.38 ms | 6.07 us | — |
| NUFFT adjoint 2D | 128x128 | torchlinops | 5.93 ms | 179.15 us | 5.92 ms | 15.32 us | — |
| NUFFT adjoint 2D | 128x128 | torchlinops (linop) | 5.47 ms | 26.01 us | 5.48 ms | 28.65 us | — |
| NUFFT adjoint 2D | 128x128 | sigpy | 9.14 ms | 283.94 us | 9.12 ms | 94.33 us | — |
| NUFFT adjoint 2D | 256x256 | torchlinops | 30.49 ms | 272.72 us | 30.46 ms | 107.17 us | — |
| NUFFT adjoint 2D | 256x256 | torchlinops (linop) | 28.52 ms | 204.21 us | 28.51 ms | 92.27 us | — |
| NUFFT adjoint 2D | 256x256 | sigpy | 36.76 ms | 1.94 ms | 36.63 ms | 329.20 us | — |


## NUFFT 3D

| Operation | Size | Library | Mean | Data Gen | Median | IQR | Peak Memory |
|-----------|------|---------|------|----------|--------|-----|-------------|
| NUFFT forward 3D | 32x32x32 | torchlinops | 8.70 ms | 198.36 us | 8.68 ms | 39.00 us | — |
| NUFFT forward 3D | 32x32x32 | torchlinops (linop) | 8.14 ms | 25.71 us | 8.14 ms | 54.55 us | — |
| NUFFT forward 3D | 32x32x32 | sigpy | 9.90 ms | 319.90 us | 9.89 ms | 19.14 us | — |
| NUFFT forward 3D | 64x64x64 | torchlinops | 96.05 ms | 1.40 ms | 95.98 ms | 478.23 us | — |
| NUFFT forward 3D | 64x64x64 | torchlinops (linop) | 94.78 ms | 211.55 us | 94.32 ms | 2.33 ms | — |
| NUFFT forward 3D | 64x64x64 | sigpy | 80.89 ms | 2.97 ms | 80.79 ms | 207.16 us | — |
| NUFFT forward 3D | 128x128x128 | torchlinops | 712.88 ms | 3.05 ms | 709.66 ms | 11.90 ms | — |
| NUFFT forward 3D | 128x128x128 | torchlinops (linop) | 688.63 ms | 3.80 ms | 687.50 ms | 22.29 ms | — |
| NUFFT forward 3D | 128x128x128 | sigpy | 769.86 ms | 51.96 ms | 769.59 ms | 2.83 ms | — |
| NUFFT adjoint 3D | 32x32x32 | torchlinops | 9.87 ms | 34.63 us | 9.85 ms | 104.25 us | — |
| NUFFT adjoint 3D | 32x32x32 | torchlinops (linop) | 9.51 ms | 17.27 us | 9.52 ms | 21.28 us | — |
| NUFFT adjoint 3D | 32x32x32 | sigpy | 10.25 ms | 163.13 us | 10.22 ms | 69.58 us | — |
| NUFFT adjoint 3D | 64x64x64 | torchlinops | 112.22 ms | 364.26 us | 110.42 ms | 8.53 ms | — |
| NUFFT adjoint 3D | 64x64x64 | torchlinops (linop) | 108.15 ms | 25.80 us | 107.91 ms | 4.70 ms | — |
| NUFFT adjoint 3D | 64x64x64 | sigpy | 83.02 ms | 225.08 us | 82.97 ms | 632.84 us | — |
| NUFFT adjoint 3D | 128x128x128 | torchlinops | 914.71 ms | 2.46 ms | 911.57 ms | 12.33 ms | — |
| NUFFT adjoint 3D | 128x128x128 | torchlinops (linop) | 885.38 ms | 211.57 us | 882.77 ms | 17.78 ms | — |
| NUFFT adjoint 3D | 128x128x128 | sigpy | 840.67 ms | 3.75 ms | 840.45 ms | 1.13 ms | — |


## Bar Charts

### Small

![Timing bar chart (small)](assets/timing_cpu_small.png)

### Medium

![Timing bar chart (medium)](assets/timing_cpu_medium.png)

### Large

![Timing bar chart (large)](assets/timing_cpu_large.png)

## Scaling Curves

![Timing scaling](assets/scaling_time_cpu.png)
