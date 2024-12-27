# Unfold/Fold Tests and Benchmarks
Copied from [triton-unfold](https://github.com/nishi951/triton-unfold)

### Results from 12/25/2024:

#### Unfold 1d/2d/3d

```sh
(triton-unfold) (base) ➜  triton-unfold git:(main) ✗ python bench_unfold1d.py
[W1225 15:44:28.295285366 unwind.cpp:214] Warning: Unsupported unwinding pattern: Address not in range (function unwinderFor)
triton
    Mean Time: 0.054 ms
    Min Time: 0.042 ms
    Max Time: 0.369 ms
    Memory: 1536 bytes
sp
    Mean Time: 0.069 ms
    Min Time: 0.065 ms
    Max Time: 0.150 ms
    Memory: 512 bytes
(triton-unfold) (base) ➜  triton-unfold git:(main) ✗ python bench_unfold2d.py
[W1225 15:44:40.268078062 unwind.cpp:214] Warning: Unsupported unwinding pattern: Address not in range (function unwinderFor)
triton
    Mean Time: 0.078 ms
    Min Time: 0.070 ms
    Max Time: 0.406 ms
    Memory: 38068736 bytes
sp
    Mean Time: 0.126 ms
    Min Time: 0.118 ms
    Max Time: 0.259 ms
    Memory: 18874368 bytes
(triton-unfold) (base) ➜  triton-unfold git:(main) ✗ python bench_unfold3d.py
[W1225 15:45:16.947023112 unwind.cpp:214] Warning: Unsupported unwinding pattern: Address not in range (function unwinderFor)
triton
    Mean Time: 2.276 ms
    Min Time: 2.237 ms
    Max Time: 2.962 ms
    Memory: 1821250560 bytes
sp
    Mean Time: 5.805 ms
    Min Time: 5.776 ms
    Max Time: 5.967 ms
    Memory: 908066816 bytes
```

#### Fold 1d/2d/3d
```sh
(triton-unfold) (base) ➜  triton-unfold git:(main) ✗ python bench_fold1d.py
[W1225 15:45:42.587807259 unwind.cpp:214] Warning: Unsupported unwinding pattern: Address not in range (function unwinderFor)
triton
    Mean Time: 0.051 ms
    Min Time: 0.043 ms
    Max Time: 0.280 ms
    Memory: 4096 bytes
sp
    Mean Time: 0.069 ms
    Min Time: 0.066 ms
    Max Time: 0.146 ms
    Memory: 1024 bytes
(triton-unfold) (base) ➜  triton-unfold git:(main) ✗ python bench_fold2d.py
[W1225 15:46:25.034849237 unwind.cpp:214] Warning: Unsupported unwinding pattern: Address not in range (function unwinderFor)
triton
    Mean Time: 0.055 ms
    Min Time: 0.048 ms
    Max Time: 0.291 ms
    Memory: 468480 bytes
sp
    Mean Time: 0.076 ms
    Min Time: 0.072 ms
    Max Time: 0.154 ms
    Memory: 80384 bytes
(triton-unfold) (base) ➜  triton-unfold git:(main) ✗ python bench_fold3d.py
[W1225 15:47:15.681459416 unwind.cpp:214] Warning: Unsupported unwinding pattern: Address not in range (function unwinderFor)
triton
    Mean Time: 4.660 ms
    Min Time: 4.503 ms
    Max Time: 4.922 ms
    Memory: 1153433600 bytes
sp
    Mean Time: 18.587 ms
    Min Time: 18.384 ms
    Max Time: 19.132 ms
    Memory: 134217728 bytes

```


