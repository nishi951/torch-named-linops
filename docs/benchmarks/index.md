# Benchmarks

Performance benchmarks for torch-named-linops.

## Metadata

- **Date**: 2026-07-03T10:14:21.324528
- **Commit**: `c5f08c7`
- **Working tree**: dirty ([patch.diff](../../benchmarks/results/latest/patch.diff))
- **PyTorch**: 2.10.0+cu128
- **CUDA**: 12.8
- **GPU**: NVIDIA GeForce RTX 3090
- **Python**: 3.10.16
- **OS**: Linux
- **Threads**: 12

## ArrayToBlocks 3D

| Operation | Size | Library | Device | Adj. Mean | Data Gen | Mean (total) | Median | IQR | Peak Memory |
|-----------|------|---------|--------|-----------|----------|--------------|--------|-----|-------------|
| ArrayToBlocks forward | 32x32x32 | torchlinops | cuda | 117.97 us | 2.61 us | 117.97 us | 116.05 us | 4.35 us | 1.59 MB |
| ArrayToBlocks forward | 32x32x32 | torchlinops (linop) | cuda | 127.29 us | 2.57 us | 127.29 us | 125.98 us | 3.20 us | 1.59 MB |
| ArrayToBlocks forward | 32x32x32 | sigpy | cuda | 49.74 us | 23.26 us | 49.74 us | 49.50 us | 995.87 ns | 3.45 MB |
| ArrayToBlocks forward | 64x64x64 | torchlinops | cuda | 116.24 us | 2.72 us | 116.24 us | 114.85 us | 3.04 us | 16.00 MB |
| ArrayToBlocks forward | 64x64x64 | torchlinops (linop) | cuda | 128.20 us | 2.74 us | 128.20 us | 127.01 us | 4.83 us | 16.00 MB |
| ArrayToBlocks forward | 64x64x64 | sigpy | cuda | 109.71 us | 22.75 us | 109.71 us | 110.48 us | 832.50 ns | 19.70 MB |
| ArrayToBlocks forward | 128x128x128 | torchlinops | cuda | 385.53 us | 24.70 us | 385.53 us | 385.02 us | 3.07 us | 132.37 MB |
| ArrayToBlocks forward | 128x128x128 | torchlinops (linop) | cuda | 394.00 us | 24.74 us | 394.00 us | 394.24 us | 2.05 us | 132.37 MB |
| ArrayToBlocks forward | 128x128x128 | sigpy | cuda | 932.45 us | 161.03 us | 932.45 us | 920.51 us | 47.59 us | 164.89 MB |
| BlocksToArray forward | 32x32x32 | torchlinops | cuda | 99.76 us | 2.91 us | 99.76 us | 99.14 us | 2.18 us | 2.68 MB |
| BlocksToArray forward | 32x32x32 | torchlinops (linop) | cuda | 110.42 us | 2.90 us | 110.42 us | 109.57 us | 2.37 us | 2.68 MB |
| BlocksToArray forward | 32x32x32 | sigpy | cuda | 46.83 us | 23.69 us | 46.83 us | 46.62 us | 920.00 ns | 4.54 MB |
| BlocksToArray forward | 64x64x64 | torchlinops | cuda | 98.64 us | 20.06 us | 98.64 us | 98.30 us | 2.05 us | 26.37 MB |
| BlocksToArray forward | 64x64x64 | torchlinops (linop) | cuda | 108.74 us | 19.94 us | 108.74 us | 108.54 us | 2.34 us | 26.37 MB |
| BlocksToArray forward | 64x64x64 | sigpy | cuda | 158.21 us | 128.65 us | 158.21 us | 157.69 us | 1.09 us | 40.07 MB |
| BlocksToArray forward | 128x128x128 | torchlinops | cuda | 299.72 us | 154.88 us | 299.72 us | 299.01 us | 7.17 us | 232.74 MB |
| BlocksToArray forward | 128x128x128 | torchlinops (linop) | cuda | 306.36 us | 164.34 us | 306.36 us | 304.13 us | 13.31 us | 232.74 MB |
| BlocksToArray forward | 128x128x128 | sigpy | cuda | 1.38 ms | 1.15 ms | 1.38 ms | 1.29 ms | 2.44 us | 349.63 MB |


## Interpolate 2D

| Operation | Size | Library | Device | Adj. Mean | Data Gen | Mean (total) | Median | IQR | Peak Memory |
|-----------|------|---------|--------|-----------|----------|--------------|--------|-----|-------------|
| Interpolate forward 2D | 64x64 | torchlinops | cuda | 99.36 us | 25.38 us | 99.36 us | 98.21 us | 3.84 us | 160.00 KB |
| Interpolate forward 2D | 64x64 | torchlinops (linop) | cuda | 104.90 us | 2.57 us | 104.90 us | 103.74 us | 2.08 us | 96.00 KB |
| Interpolate forward 2D | 64x64 | sigpy | cuda | 176.53 us | 81.98 us | 176.53 us | 173.50 us | 7.04 us | 528.00 KB |
| Interpolate forward 2D | 128x128 | torchlinops | cuda | 129.15 us | 25.92 us | 129.15 us | 127.07 us | 2.78 us | 640.00 KB |
| Interpolate forward 2D | 128x128 | torchlinops (linop) | cuda | 135.88 us | 2.56 us | 135.88 us | 134.88 us | 2.11 us | 384.00 KB |
| Interpolate forward 2D | 128x128 | sigpy | cuda | 173.42 us | 80.91 us | 173.42 us | 173.56 us | 3.41 us | 528.00 KB |
| Interpolate forward 2D | 256x256 | torchlinops | cuda | 275.84 us | 25.60 us | 275.84 us | 275.46 us | 3.07 us | 2.50 MB |
| Interpolate forward 2D | 256x256 | torchlinops (linop) | cuda | 285.11 us | 2.85 us | 285.11 us | 284.58 us | 3.01 us | 1.50 MB |
| Interpolate forward 2D | 256x256 | sigpy | cuda | 258.40 us | 83.67 us | 258.40 us | 258.24 us | 922.98 ns | 1.77 MB |
| Interpolate adjoint 2D | 64x64 | torchlinops | cuda | 100.35 us | 25.60 us | 100.35 us | 99.01 us | 4.99 us | 160.00 KB |
| Interpolate adjoint 2D | 64x64 | torchlinops (linop) | cuda | 111.16 us | 2.61 us | 111.16 us | 110.59 us | 2.11 us | 96.00 KB |
| Interpolate adjoint 2D | 64x64 | sigpy | cuda | 174.70 us | 80.52 us | 174.70 us | 174.35 us | 4.84 us | 528.00 KB |
| Interpolate adjoint 2D | 128x128 | torchlinops | cuda | 127.59 us | 26.10 us | 127.59 us | 126.98 us | 3.65 us | 640.00 KB |
| Interpolate adjoint 2D | 128x128 | torchlinops (linop) | cuda | 133.59 us | 2.56 us | 133.59 us | 133.12 us | 3.81 us | 384.00 KB |
| Interpolate adjoint 2D | 128x128 | sigpy | cuda | 179.61 us | 78.62 us | 179.61 us | 179.39 us | 1.67 us | 528.00 KB |
| Interpolate adjoint 2D | 256x256 | torchlinops | cuda | 257.40 us | 25.97 us | 257.40 us | 256.00 us | 3.07 us | 2.50 MB |
| Interpolate adjoint 2D | 256x256 | torchlinops (linop) | cuda | 266.87 us | 2.84 us | 266.87 us | 266.24 us | 3.07 us | 1.50 MB |
| Interpolate adjoint 2D | 256x256 | sigpy | cuda | 259.23 us | 77.64 us | 259.23 us | 258.69 us | 1.26 us | 1.77 MB |


## Interpolate 3D

| Operation | Size | Library | Device | Adj. Mean | Data Gen | Mean (total) | Median | IQR | Peak Memory |
|-----------|------|---------|--------|-----------|----------|--------------|--------|-----|-------------|
| Interpolate forward 3D | 32x32x32 | torchlinops | cuda | 112.29 us | 26.32 us | 112.29 us | 111.62 us | 4.10 us | 656.00 KB |
| Interpolate forward 3D | 32x32x32 | torchlinops (linop) | cuda | 119.21 us | 2.56 us | 119.21 us | 118.78 us | 2.40 us | 560.00 KB |
| Interpolate forward 3D | 32x32x32 | sigpy | cuda | 191.16 us | 95.08 us | 191.16 us | 191.39 us | 3.71 us | 1.02 MB |
| Interpolate forward 3D | 64x64x64 | torchlinops | cuda | 343.69 us | 25.95 us | 343.69 us | 343.68 us | 3.30 us | 5.12 MB |
| Interpolate forward 3D | 64x64x64 | torchlinops (linop) | cuda | 357.82 us | 2.76 us | 357.82 us | 358.21 us | 4.45 us | 4.38 MB |
| Interpolate forward 3D | 64x64x64 | sigpy | cuda | 564.17 us | 93.61 us | 564.17 us | 562.37 us | 4.13 us | 6.52 MB |
| Interpolate forward 3D | 128x128x128 | torchlinops | cuda | 2.28 ms | 27.19 us | 2.28 ms | 2.28 ms | 13.31 us | 42.00 MB |
| Interpolate forward 3D | 128x128x128 | torchlinops (linop) | cuda | 2.61 ms | 25.02 us | 2.61 ms | 2.63 ms | 108.54 us | 35.00 MB |
| Interpolate forward 3D | 128x128x128 | sigpy | cuda | 3.38 ms | 204.25 us | 3.38 ms | 3.35 ms | 79.72 us | 48.52 MB |
| Interpolate adjoint 3D | 32x32x32 | torchlinops | cuda | 111.98 us | 26.06 us | 111.98 us | 110.59 us | 4.03 us | 336.00 KB |
| Interpolate adjoint 3D | 32x32x32 | torchlinops (linop) | cuda | 118.89 us | 2.54 us | 118.89 us | 117.76 us | 2.30 us | 336.00 KB |
| Interpolate adjoint 3D | 32x32x32 | sigpy | cuda | 191.38 us | 91.57 us | 191.38 us | 191.01 us | 3.56 us | 528.00 KB |
| Interpolate adjoint 3D | 64x64x64 | torchlinops | cuda | 327.56 us | 26.21 us | 327.56 us | 327.63 us | 2.34 us | 2.62 MB |
| Interpolate adjoint 3D | 64x64x64 | torchlinops (linop) | cuda | 339.64 us | 2.53 us | 339.64 us | 340.99 us | 5.12 us | 2.62 MB |
| Interpolate adjoint 3D | 64x64x64 | sigpy | cuda | 555.22 us | 90.76 us | 555.22 us | 554.88 us | 3.62 us | 4.77 MB |
| Interpolate adjoint 3D | 128x128x128 | torchlinops | cuda | 2.24 ms | 29.40 us | 2.24 ms | 2.24 ms | 8.19 us | 21.00 MB |
| Interpolate adjoint 3D | 128x128x128 | torchlinops (linop) | cuda | 2.30 ms | 2.83 us | 2.30 ms | 2.31 ms | 18.43 us | 21.00 MB |
| Interpolate adjoint 3D | 128x128x128 | sigpy | cuda | 3.30 ms | 92.66 us | 3.30 ms | 3.28 ms | 17.20 us | 22.52 MB |


## NUFFT 2D

| Operation | Size | Library | Device | Adj. Mean | Data Gen | Mean (total) | Median | IQR | Peak Memory |
|-----------|------|---------|--------|-----------|----------|--------------|--------|-----|-------------|
| NUFFT forward 2D | 64x64 | torchlinops | cuda | 456.99 us | 28.11 us | 456.99 us | 454.66 us | 15.36 us | 312.00 KB |
| NUFFT forward 2D | 64x64 | torchlinops (linop) | cuda | 276.79 us | 2.52 us | 276.79 us | 275.39 us | 10.56 us | 312.50 KB |
| NUFFT forward 2D | 64x64 | sigpy | cuda | 625.25 us | 107.17 us | 625.25 us | 623.43 us | 7.45 us | 262.00 KB |
| NUFFT forward 2D | 128x128 | torchlinops | cuda | 580.21 us | 28.23 us | 580.21 us | 578.56 us | 10.24 us | 1.22 MB |
| NUFFT forward 2D | 128x128 | torchlinops (linop) | cuda | 283.04 us | 2.47 us | 283.04 us | 281.94 us | 9.22 us | 1.22 MB |
| NUFFT forward 2D | 128x128 | sigpy | cuda | 701.23 us | 106.26 us | 701.23 us | 701.16 us | 4.43 us | 1000.00 KB |
| NUFFT forward 2D | 256x256 | torchlinops | cuda | 1.02 ms | 28.21 us | 1.02 ms | 1.01 ms | 19.46 us | 4.88 MB |
| NUFFT forward 2D | 256x256 | torchlinops (linop) | cuda | 428.24 us | 2.83 us | 428.24 us | 427.90 us | 9.98 us | 4.88 MB |
| NUFFT forward 2D | 256x256 | sigpy | cuda | 1.15 ms | 107.42 us | 1.15 ms | 1.14 ms | 10.30 us | 3.86 MB |
| NUFFT adjoint 2D | 64x64 | torchlinops | cuda | 443.49 us | 28.17 us | 443.49 us | 442.62 us | 11.20 us | 312.00 KB |
| NUFFT adjoint 2D | 64x64 | torchlinops (linop) | cuda | 261.23 us | 2.45 us | 261.23 us | 260.06 us | 10.24 us | 312.50 KB |
| NUFFT adjoint 2D | 64x64 | sigpy | cuda | 615.37 us | 104.97 us | 615.37 us | 615.94 us | 3.48 us | 262.00 KB |
| NUFFT adjoint 2D | 128x128 | torchlinops | cuda | 558.77 us | 27.97 us | 558.77 us | 558.08 us | 12.19 us | 1.22 MB |
| NUFFT adjoint 2D | 128x128 | torchlinops (linop) | cuda | 258.88 us | 2.45 us | 258.88 us | 257.06 us | 8.29 us | 1.22 MB |
| NUFFT adjoint 2D | 128x128 | sigpy | cuda | 615.11 us | 103.07 us | 615.11 us | 612.61 us | 4.39 us | 1000.00 KB |
| NUFFT adjoint 2D | 256x256 | torchlinops | cuda | 960.78 us | 28.19 us | 960.78 us | 958.40 us | 9.22 us | 4.88 MB |
| NUFFT adjoint 2D | 256x256 | torchlinops (linop) | cuda | 353.55 us | 2.81 us | 353.55 us | 353.15 us | 6.14 us | 4.88 MB |
| NUFFT adjoint 2D | 256x256 | sigpy | cuda | 909.72 us | 104.23 us | 909.72 us | 908.98 us | 3.78 us | 3.86 MB |


## NUFFT 3D

| Operation | Size | Library | Device | Adj. Mean | Data Gen | Mean (total) | Median | IQR | Peak Memory |
|-----------|------|---------|--------|-----------|----------|--------------|--------|-----|-------------|
| NUFFT forward 3D | 32x32x32 | torchlinops | cuda | 886.79 us | 23.34 us | 886.79 us | 886.64 us | 29.89 us | 2.91 MB |
| NUFFT forward 3D | 32x32x32 | torchlinops (linop) | cuda | 282.18 us | 2.55 us | 282.18 us | 281.60 us | 10.24 us | 2.91 MB |
| NUFFT forward 3D | 32x32x32 | sigpy | cuda | 1.08 ms | 121.20 us | 1.08 ms | 1.06 ms | 10.83 us | 2.23 MB |
| NUFFT forward 3D | 64x64x64 | torchlinops | cuda | 3.24 ms | 23.23 us | 3.24 ms | 3.25 ms | 225.28 us | 23.66 MB |
| NUFFT forward 3D | 64x64x64 | torchlinops (linop) | cuda | 503.23 us | 2.71 us | 503.23 us | 502.78 us | 11.07 us | 23.66 MB |
| NUFFT forward 3D | 64x64x64 | sigpy | cuda | 2.16 ms | 120.51 us | 2.16 ms | 2.15 ms | 12.77 us | 18.23 MB |
| NUFFT forward 3D | 128x128x128 | torchlinops | cuda | 34.79 ms | 33.02 us | 34.79 ms | 34.72 ms | 370.69 us | 191.00 MB |
| NUFFT forward 3D | 128x128x128 | torchlinops (linop) | cuda | 3.49 ms | 24.73 us | 3.49 ms | 3.47 ms | 74.75 us | 190.00 MB |
| NUFFT forward 3D | 128x128x128 | sigpy | cuda | 10.56 ms | 237.79 us | 10.56 ms | 10.47 ms | 15.55 us | 142.27 MB |
| NUFFT adjoint 3D | 32x32x32 | torchlinops | cuda | 851.62 us | 23.32 us | 851.62 us | 849.47 us | 11.49 us | 2.69 MB |
| NUFFT adjoint 3D | 32x32x32 | torchlinops (linop) | cuda | 274.78 us | 2.47 us | 274.78 us | 273.41 us | 9.28 us | 2.69 MB |
| NUFFT adjoint 3D | 32x32x32 | sigpy | cuda | 770.85 us | 116.82 us | 770.85 us | 769.15 us | 3.85 us | 1.98 MB |
| NUFFT adjoint 3D | 64x64x64 | torchlinops | cuda | 1.89 ms | 23.37 us | 1.89 ms | 1.89 ms | 12.29 us | 21.91 MB |
| NUFFT adjoint 3D | 64x64x64 | torchlinops (linop) | cuda | 487.23 us | 2.47 us | 487.23 us | 487.42 us | 6.14 us | 21.91 MB |
| NUFFT adjoint 3D | 64x64x64 | sigpy | cuda | 1.87 ms | 117.35 us | 1.87 ms | 1.86 ms | 26.37 us | 12.86 MB |
| NUFFT adjoint 3D | 128x128x128 | torchlinops | cuda | 35.32 ms | 27.49 us | 35.32 ms | 35.23 ms | 274.27 us | 176.00 MB |
| NUFFT adjoint 3D | 128x128x128 | torchlinops (linop) | cuda | 3.43 ms | 2.72 us | 3.43 ms | 3.43 ms | 55.10 us | 176.00 MB |
| NUFFT adjoint 3D | 128x128x128 | sigpy | cuda | 10.56 ms | 127.13 us | 10.56 ms | 10.47 ms | 31.92 us | 103.27 MB |


## Bar Charts

### Small

![Timing bar chart (small)](assets/timing_small.png)

### Medium

![Timing bar chart (medium)](assets/timing_medium.png)

### Large

![Timing bar chart (large)](assets/timing_large.png)

## Scaling Curves

![Timing scaling](assets/scaling_time.png)

![Memory scaling](assets/scaling_memory.png)
