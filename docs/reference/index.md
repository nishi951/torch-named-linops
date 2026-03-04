# API Reference

The API reference is organized into the following sections. Each section links to
detailed, auto-generated documentation derived from the source code docstrings.

## [Linops](linops/index.md)

The core linear operator classes. Start with [NamedLinop](linops/namedlinop.md)
for the base class, then explore specific operators:

- **Core**: [Dense](linops/dense.md), [Diagonal & Scalar](linops/diagonal.md), [Identity](linops/identity.md), [FFT](linops/fft.md)
- **Composition**: [Chain](linops/chain.md), [Add](linops/add.md), [Concat & Stack](linops/concat_stack.md)
- **Transforms**: [NUFFT](linops/nufft.md), [Interpolation](linops/interp.md), [Sampling](linops/sampling.md)
- **Structural**: [Einops](linops/einops.md), [Padding & Truncation](linops/pad.md), [ArrayToBlocks](linops/array_manipulation.md)
- **Device & Batching**: [Device Transfer](linops/device.md), [Batching](linops/batch.md), [Splitting](linops/split.md)

## [Functional](functional/index.md)

Low-level functional interfaces that operate directly on tensors:

- [Interpolation](functional/interp.md) -- grid/ungrid operations
- [NUFFT](functional/nufft.md) -- non-uniform FFT
- [Padding](functional/pad.md) -- centered padding and cropping
- [Unfold/Fold](functional/unfold.md) -- block extraction and reassembly

## [Named Dimensions](nameddim/nameddimension.md)

The dimension-naming abstraction that underpins the library:

- [NamedDimension & Dim](nameddim/nameddimension.md) -- the core dimension type
- [NamedShape](nameddim/namedshape.md) -- paired input/output shape specs
- [Matching](nameddim/matching.md) -- shape comparison and partitioning

## [Algorithms](alg/cg.md)

Iterative solvers and related tools:

- [Conjugate Gradient](alg/cg.md)
- [Power Method](alg/powermethod.md)
- [Polynomial Preconditioner](alg/poly.md)

## [Utilities](utils/math.md)

Helper functions for math, memory management, and benchmarking:

- [Math & FFT](utils/math.md) -- centered FFT, adjoint testing
- [Memory & Device](utils/memory.md) -- device transfer, memory reporting
- [Benchmarking](utils/benchmark.md) -- GPU/CPU benchmarking
