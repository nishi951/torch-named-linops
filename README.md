# Torch LinOp
A selection of linop abstractions implemented in PyTorch.

Attempts to be matrix-free.

Unrelated to the (also good) [torch_linops](https://github.com/cvxgrp/torch_linops)

Support for fast normal operators and batching across matrix dimensions


# Features
- Full support for complex numbers
  - Adjoint takes the conjugate transpose
- Linear Operator Types
  - Diagonal: Elementwise multiply
  - Dense: Regular matrix multiplication
  - FFT: Fourier Transform
  - NUFFT: Non Uniform Fourier Transform
  - Stack, Split: Stacking and splitting linear operators along various dimensions
- Named Dimensions
  - Name the input and output dimensions of the linear operators
  - Enables consolidation/automatic simplification.
  - Graph the computation with names
- Slicing Linear Operators
  - Related to Split - Allows for batching the computation in arbitrary ways
    - Saves GPU memory
- Simplify Chains of Linear Operators
  - For example, Split followed by Stack simplifies to Identity
    - ...if the splitting and stacking is done along the same dimension.
- Functional interface
  - For when the linear operators need to change e.g. each batch.
- Batching over multiple linops
  - For example, if a dimension is expanded and later combined.

# Implementation
- Data structures
  - The linop data
    - Stores the actual data for the linop
    - Basically just a dictionary
  - The linop ordering
  - The linop execution strategy (schedule)
    - How the computation is actually batched and executed
      - Include the batching strategy for each input/output dim

## Batching strategy
- BatchSplit and BatchCombine
  - Beginning of the batching is a `split`
    - End of the batching is a `combine`
  - If a linop's input is batched, it's a horizontal split
    - Needs a sum combine
  - If a linop's output is batched, it's a vertical split
    - Needs a stack combine


# TODO
- LinOp
  - forward
  - adjoint
  - normal

