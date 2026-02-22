# Why this package?

## The problem with linops

Linear operators are fundamental to many areas of scientific computing -- optimization, inverse problems, signal processing, and deep learning all rely heavily on them. Yet working with linops in standard numerical frameworks presents several persistent challenges:

### Shape ambiguity

Standard matrix-vector operations in PyTorch work on flat or positionally-indexed tensors. The meaning of each axis is implicit and must be tracked mentally or via comments:

```python
# What does dimension 0 mean? 1? 2?
# Is this (batch, channel, spatial) or (spatial_x, spatial_y, coil)?
y = A @ x.reshape(-1)
y = y.reshape(batch, nx, ny)
```

This positional indexing is fragile. Reordering dimensions, adding batch dimensions, or composing multiple operators requires careful bookkeeping that the framework does not help with.

### Manual adjoint derivation

Implementing the adjoint (conjugate transpose) of a complex linear operator is tedious and error-prone. For a simple dense matrix, the adjoint is just `.conj().T`. But for a composition of FFTs, interpolation, sampling masks, and sensitivity maps, the adjoint must be manually derived and tested for each combination. A sign error or transposition bug in the adjoint can silently produce wrong results in iterative algorithms.

### Composition difficulty

Composing multiple operators (e.g., $A = S \circ F \circ C$ for MRI reconstruction, where $S$ is sampling, $F$ is FFT, and $C$ is coil sensitivity) requires manually verifying that intermediate shapes match at each stage. This verification is typically done at runtime via shape errors, which are cryptic and hard to debug.

### Multi-GPU complexity

Distributing a large linear operator across multiple GPUs typically requires writing custom parallel code: manually splitting data, managing device transfers, synchronizing streams, and reassembling results. This code is orthogonal to the mathematical definition of the operator and must be rewritten for each new operator or hardware configuration.

## The solution: named dimensions

`torch-named-linops` addresses these issues by making **named dimensions** first-class citizens in the definition of linear operators.

### Self-documenting shapes

Instead of positional indices, each operator declares its input and output shapes using meaningful names:

```python
F = FFT(ndim=2, grid_shapes=(Dim("NxNy"), Dim("KxKy")))
# Shape: (Nx, Ny) -> (Kx, Ky)
```

The shape specification is part of the operator's identity. Reading `(Nx, Ny) -> (Kx, Ky)` immediately communicates that this is a 2D Fourier transform from image space to k-space. No comments needed.

### Automatic adjoints and normals

The library provides a framework where the adjoint $A^H$ is automatically constructed by swapping the forward and adjoint functions and flipping the shape:

```python
A.H          # Adjoint: (Kx, Ky) -> (Nx, Ny)
A.H.H        # Back to original: (Nx, Ny) -> (Kx, Ky)
A.N          # Normal A^H A: (Nx, Ny) -> (Nx, Ny)
```

Many operators also provide optimized normal implementations. For example, the FFT's normal is the identity (since $F^H F = I$ with orthonormal normalization), and a diagonal operator's normal is simply $|w|^2$.

### Compositionality

Operators compose naturally using Python's `@` operator, with automatic shape validation:

```python
A = S @ F @ C   # Composition: apply C, then F, then S
# Shape checking happens at construction time
A.H             # Adjoint of the chain: C^H @ F^H @ S^H
A.N             # Normal: optimized A^H A
```

The `Chain` operator verifies at construction time that each linop's output shape matches the next linop's input shape, catching shape errors before any data flows through the system.

### Splitting and distributed computing

Because dimensions are named, operators can be split along specific dimensions:

```python
from torchlinops.linops.split import create_batched_linop, BatchSpec

spec = BatchSpec(
    batch_sizes={"Nx": 128},
    device_matrix=[torch.device("cuda:0"), torch.device("cuda:1")],
    base_device=torch.device("cuda:0"),
)
A_distributed = create_batched_linop(A, spec)
# A_distributed behaves like A but runs across two GPUs
```

The splitting, device transfer, and reassembly logic is handled by the library. Each operator only needs to implement `split_forward` to describe how it decomposes along its dimensions. See [Multi-GPU Splitting](multi_gpu.md) for the full details.
