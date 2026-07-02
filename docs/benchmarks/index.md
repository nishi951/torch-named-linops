# Benchmarks

Performance benchmarks for torch-named-linops.

## Metadata

- **Date**: 2026-07-01T23:52:35.810692
- **Commit**: `9e13cbe`
- **Working tree**: clean
- **PyTorch**: 2.10.0+cu128
- **CUDA**: 12.8
- **GPU**: NVIDIA GeForce RTX 3090
- **Python**: 3.10.16
- **OS**: Linux
- **Threads**: 12

## ArrayToBlocks

| Operation | Device | Mean | Median | IQR | Peak Memory |
|-----------|--------|------|--------|-----|-------------|
| ArrayToBlocks forward | cuda | 101.49 us | 101.49 us | 0.00 ns | 1.90 MB |
| BlocksToArray forward | cuda | 86.54 us | 86.54 us | 0.00 ns | 1.90 MB |


## Interpolate

| Operation | Device | Mean | Median | IQR | Peak Memory |
|-----------|--------|------|--------|-----|-------------|
| Interpolate forward 2D | cuda | 80.80 us | 80.80 us | 0.00 ns | 64.00 KB |
| Interpolate adjoint 2D | cuda | 148.56 us | 148.42 us | 1.61 us | 96.00 KB |
| Interpolate forward 3D | cuda | 82.22 us | 82.22 us | 0.00 ns | 276.00 KB |
| Interpolate adjoint 3D | cuda | 149.59 us | 149.53 us | 529.73 ns | 532.00 KB |


## NUFFT

| Operation | Device | Mean | Median | IQR | Peak Memory |
|-----------|--------|------|--------|-----|-------------|
| NUFFT forward 2D | cuda | 316.83 us | 259.66 us | 16.54 us | 264.50 KB |
| NUFFT adjoint 2D | cuda | 702.44 us | 702.44 us | 0.00 ns | 272.50 KB |
| NUFFT forward 3D | cuda | 258.10 us | 258.10 us | 171.41 ns | 2.83 MB |
| NUFFT adjoint 3D | cuda | 761.54 us | 761.54 us | 0.00 ns | 2.83 MB |


## Charts

![Timing comparison](assets/timing_comparison.png)

![Memory comparison](assets/memory_comparison.png)
