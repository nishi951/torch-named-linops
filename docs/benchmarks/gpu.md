# GPU Benchmarks

Performance benchmarks on GPU.

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
| ArrayToBlocks forward | 32x32x32 | torchlinops | 117.34 us | 116.48 us | 3.17 us | 17.50 MB |
| ArrayToBlocks forward | 32x32x32 | torchlinops (linop) | 128.34 us | 127.79 us | 3.90 us | 17.50 MB |
| ArrayToBlocks forward | 32x32x32 | sigpy | 43.56 us | 43.01 us | 1.79 us | 2.86 MB |
| ArrayToBlocks forward | 64x64x64 | torchlinops | 116.08 us | 115.62 us | 2.58 us | 30.43 MB |
| ArrayToBlocks forward | 64x64x64 | torchlinops (linop) | 127.05 us | 125.95 us | 3.07 us | 30.43 MB |
| ArrayToBlocks forward | 64x64x64 | sigpy | 116.28 us | 115.71 us | 1.99 us | 21.70 MB |
| ArrayToBlocks forward | 128x128x128 | torchlinops | 391.60 us | 391.17 us | 2.05 us | 147.62 MB |
| ArrayToBlocks forward | 128x128x128 | torchlinops (linop) | 400.59 us | 400.38 us | 2.05 us | 147.62 MB |
| ArrayToBlocks forward | 128x128x128 | sigpy | 758.66 us | 760.83 us | 4.12 us | 180.89 MB |
| BlocksToArray forward | 32x32x32 | torchlinops | 98.89 us | 98.30 us | 2.37 us | 18.59 MB |
| BlocksToArray forward | 32x32x32 | torchlinops (linop) | 112.49 us | 111.78 us | 3.04 us | 18.59 MB |
| BlocksToArray forward | 32x32x32 | sigpy | 38.01 us | 37.89 us | 1.55 us | 8.52 MB |
| BlocksToArray forward | 64x64x64 | torchlinops | 103.05 us | 102.40 us | 2.05 us | 41.62 MB |
| BlocksToArray forward | 64x64x64 | torchlinops (linop) | 114.49 us | 112.64 us | 4.10 us | 41.62 MB |
| BlocksToArray forward | 64x64x64 | sigpy | 32.69 us | 32.77 us | 672.00 ns | 53.25 MB |
| BlocksToArray forward | 128x128x128 | torchlinops | 631.23 us | 523.26 us | 273.66 us | 247.99 MB |
| BlocksToArray forward | 128x128x128 | torchlinops (linop) | 330.35 us | 316.42 us | 23.55 us | 247.99 MB |
| BlocksToArray forward | 128x128x128 | sigpy | 220.31 us | 218.32 us | 2.05 us | 466.00 MB |


## Interpolate 2D

| Operation | Size | Library | Mean | Median | IQR | Peak Memory |
|-----------|------|---------|------|--------|-----|-------------|
| Interpolate forward 2D | 64x64 | torchlinops | 93.31 us | 92.16 us | 3.07 us | 15.34 MB |
| Interpolate forward 2D | 64x64 | torchlinops (linop) | 106.60 us | 105.47 us | 2.96 us | 15.34 MB |
| Interpolate forward 2D | 64x64 | sigpy | 100.59 us | 100.32 us | 1.86 us | 528.00 KB |
| Interpolate forward 2D | 128x128 | torchlinops | 124.97 us | 123.90 us | 1.92 us | 15.63 MB |
| Interpolate forward 2D | 128x128 | torchlinops (linop) | 136.21 us | 135.94 us | 2.06 us | 15.63 MB |
| Interpolate forward 2D | 128x128 | sigpy | 128.55 us | 127.28 us | 1.98 us | 656.00 KB |
| Interpolate forward 2D | 256x256 | torchlinops | 298.10 us | 296.85 us | 2.05 us | 16.75 MB |
| Interpolate forward 2D | 256x256 | torchlinops (linop) | 306.08 us | 308.13 us | 2.11 us | 16.75 MB |
| Interpolate forward 2D | 256x256 | sigpy | 278.87 us | 276.48 us | 4.34 us | 2.27 MB |
| Interpolate adjoint 2D | 64x64 | torchlinops | 93.51 us | 92.21 us | 2.74 us | 15.34 MB |
| Interpolate adjoint 2D | 64x64 | torchlinops (linop) | 107.14 us | 106.50 us | 2.11 us | 15.34 MB |
| Interpolate adjoint 2D | 64x64 | sigpy | 100.02 us | 99.33 us | 1.25 us | 784.00 KB |
| Interpolate adjoint 2D | 128x128 | torchlinops | 121.50 us | 120.83 us | 1.25 us | 15.66 MB |
| Interpolate adjoint 2D | 128x128 | torchlinops (linop) | 134.14 us | 133.84 us | 2.17 us | 15.66 MB |
| Interpolate adjoint 2D | 128x128 | sigpy | 127.73 us | 126.98 us | 1.15 us | 784.00 KB |
| Interpolate adjoint 2D | 256x256 | torchlinops | 258.76 us | 257.02 us | 2.24 us | 16.91 MB |
| Interpolate adjoint 2D | 256x256 | torchlinops (linop) | 270.12 us | 269.31 us | 3.06 us | 16.91 MB |
| Interpolate adjoint 2D | 256x256 | sigpy | 280.06 us | 279.44 us | 2.98 us | 2.27 MB |


## Interpolate 3D

| Operation | Size | Library | Mean | Median | IQR | Peak Memory |
|-----------|------|---------|------|--------|-----|-------------|
| Interpolate forward 3D | 32x32x32 | torchlinops | 107.54 us | 107.23 us | 1.76 us | 16.45 MB |
| Interpolate forward 3D | 32x32x32 | torchlinops (linop) | 120.18 us | 119.81 us | 2.05 us | 16.45 MB |
| Interpolate forward 3D | 32x32x32 | sigpy | 206.57 us | 205.06 us | 2.75 us | 1.27 MB |
| Interpolate forward 3D | 64x64x64 | torchlinops | 372.72 us | 371.71 us | 2.05 us | 20.28 MB |
| Interpolate forward 3D | 64x64x64 | torchlinops (linop) | 384.67 us | 384.00 us | 3.77 us | 20.28 MB |
| Interpolate forward 3D | 64x64x64 | sigpy | 575.54 us | 568.32 us | 9.22 us | 8.52 MB |
| Interpolate forward 3D | 128x128x128 | torchlinops | 2.92 ms | 2.78 ms | 327.42 us | 51.28 MB |
| Interpolate forward 3D | 128x128x128 | torchlinops (linop) | 2.51 ms | 2.50 ms | 6.30 us | 51.28 MB |
| Interpolate forward 3D | 128x128x128 | sigpy | 3.51 ms | 3.51 ms | 959.99 ns | 64.52 MB |
| Interpolate adjoint 3D | 32x32x32 | torchlinops | 107.72 us | 107.42 us | 2.24 us | 15.61 MB |
| Interpolate adjoint 3D | 32x32x32 | torchlinops (linop) | 116.93 us | 116.74 us | 2.05 us | 15.61 MB |
| Interpolate adjoint 3D | 32x32x32 | sigpy | 200.19 us | 199.68 us | 2.05 us | 528.00 KB |
| Interpolate adjoint 3D | 64x64x64 | torchlinops | 352.03 us | 351.04 us | 2.08 us | 17.95 MB |
| Interpolate adjoint 3D | 64x64x64 | torchlinops (linop) | 363.67 us | 363.52 us | 2.26 us | 17.95 MB |
| Interpolate adjoint 3D | 64x64x64 | sigpy | 571.12 us | 565.33 us | 15.36 us | 3.14 MB |
| Interpolate adjoint 3D | 128x128x128 | torchlinops | 2.82 ms | 2.75 ms | 185.34 us | 37.70 MB |
| Interpolate adjoint 3D | 128x128x128 | torchlinops (linop) | 2.55 ms | 2.54 ms | 5.12 us | 37.70 MB |
| Interpolate adjoint 3D | 128x128x128 | sigpy | 3.67 ms | 3.62 ms | 12.46 us | 24.52 MB |


## NUFFT 2D

| Operation | Size | Library | Mean | Median | IQR | Peak Memory |
|-----------|------|---------|------|--------|-----|-------------|
| NUFFT forward 2D | 64x64 | torchlinops | 555.75 us | 535.55 us | 19.63 us | 360.50 KB |
| NUFFT forward 2D | 64x64 | torchlinops (linop) | 348.67 us | 345.09 us | 15.25 us | 312.50 KB |
| NUFFT forward 2D | 64x64 | sigpy | 523.23 us | 521.23 us | 13.70 us | 294.00 KB |
| NUFFT forward 2D | 128x128 | torchlinops | 652.85 us | 650.24 us | 24.85 us | 1.41 MB |
| NUFFT forward 2D | 128x128 | torchlinops (linop) | 349.19 us | 347.36 us | 12.18 us | 1.22 MB |
| NUFFT forward 2D | 128x128 | sigpy | 631.24 us | 626.69 us | 16.45 us | 1.10 MB |
| NUFFT forward 2D | 256x256 | torchlinops | 1.66 ms | 1.80 ms | 149.74 us | 5.63 MB |
| NUFFT forward 2D | 256x256 | torchlinops (linop) | 507.52 us | 497.68 us | 15.36 us | 4.88 MB |
| NUFFT forward 2D | 256x256 | sigpy | 1.16 ms | 1.15 ms | 15.52 us | 4.36 MB |
| NUFFT adjoint 2D | 64x64 | torchlinops | 522.50 us | 515.07 us | 18.89 us | 360.50 KB |
| NUFFT adjoint 2D | 64x64 | torchlinops (linop) | 327.00 us | 325.63 us | 11.32 us | 312.50 KB |
| NUFFT adjoint 2D | 64x64 | sigpy | 455.18 us | 452.70 us | 14.34 us | 272.00 KB |
| NUFFT adjoint 2D | 128x128 | torchlinops | 626.58 us | 614.40 us | 30.18 us | 1.49 MB |
| NUFFT adjoint 2D | 128x128 | torchlinops (linop) | 326.81 us | 323.15 us | 11.54 us | 1.30 MB |
| NUFFT adjoint 2D | 128x128 | sigpy | 449.42 us | 449.47 us | 10.46 us | 936.00 KB |
| NUFFT adjoint 2D | 256x256 | torchlinops | 955.80 us | 953.34 us | 10.24 us | 5.63 MB |
| NUFFT adjoint 2D | 256x256 | torchlinops (linop) | 384.11 us | 375.55 us | 29.70 us | 4.88 MB |
| NUFFT adjoint 2D | 256x256 | sigpy | 935.01 us | 931.84 us | 4.03 us | 4.36 MB |


## NUFFT 3D

| Operation | Size | Library | Mean | Median | IQR | Peak Memory |
|-----------|------|---------|------|--------|-----|-------------|
| NUFFT forward 3D | 32x32x32 | torchlinops | 922.10 us | 921.73 us | 12.19 us | 4.33 MB |
| NUFFT forward 3D | 32x32x32 | torchlinops (linop) | 349.38 us | 347.07 us | 13.31 us | 4.16 MB |
| NUFFT forward 3D | 32x32x32 | sigpy | 1.02 ms | 1.02 ms | 17.92 us | 2.24 MB |
| NUFFT forward 3D | 64x64x64 | torchlinops | 2.07 ms | 2.06 ms | 9.47 us | 26.28 MB |
| NUFFT forward 3D | 64x64x64 | torchlinops (linop) | 593.28 us | 591.04 us | 12.19 us | 24.91 MB |
| NUFFT forward 3D | 64x64x64 | sigpy | 2.24 ms | 2.22 ms | 18.22 us | 20.23 MB |
| NUFFT forward 3D | 128x128x128 | torchlinops | 28.52 ms | 28.36 ms | 350.46 us | 203.25 MB |
| NUFFT forward 3D | 128x128x128 | torchlinops (linop) | 3.30 ms | 3.30 ms | 5.84 us | 191.25 MB |
| NUFFT forward 3D | 128x128x128 | sigpy | 10.35 ms | 10.19 ms | 4.31 us | 158.27 MB |
| NUFFT adjoint 3D | 32x32x32 | torchlinops | 736.48 us | 735.10 us | 13.66 us | 4.11 MB |
| NUFFT adjoint 3D | 32x32x32 | torchlinops (linop) | 336.46 us | 334.46 us | 13.48 us | 3.94 MB |
| NUFFT adjoint 3D | 32x32x32 | sigpy | 639.97 us | 636.93 us | 6.05 us | 1.98 MB |
| NUFFT adjoint 3D | 64x64x64 | torchlinops | 2.04 ms | 2.04 ms | 6.58 us | 24.75 MB |
| NUFFT adjoint 3D | 64x64x64 | torchlinops (linop) | 537.43 us | 535.71 us | 8.21 us | 23.38 MB |
| NUFFT adjoint 3D | 64x64x64 | sigpy | 1.99 ms | 1.98 ms | 6.91 us | 12.97 MB |
| NUFFT adjoint 3D | 128x128x128 | torchlinops | 34.52 ms | 34.53 ms | 220.44 us | 188.25 MB |
| NUFFT adjoint 3D | 128x128x128 | torchlinops (linop) | 3.45 ms | 3.45 ms | 10.51 us | 177.25 MB |
| NUFFT adjoint 3D | 128x128x128 | sigpy | 11.20 ms | 11.58 ms | 984.12 us | 103.17 MB |


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
