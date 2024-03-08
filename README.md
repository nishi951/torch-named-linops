# Torchlinops
A library for fast MRI reconstruction and prototyping.

Includes:

- A selection of linop abstractions implemented in PyTorch.
  - Supports named dimensions and slicing/batching over them.
- A set of `app`s that encapsulate common recon or prototyping tasks.

Note: this project is Unrelated to the (also good) [torch_linops](https://github.com/cvxgrp/torch_linops)

## API
### Modules
Module summary:
- `torchlinops` : Base module with all linops
  - `.app` : Special "apps" for convenient wrapping of lower-level functionality
  - `.mri` : Submodule with mri-specific functionality
    - `.app` : MRI-specific apps
    - `.data` : Data processing functions
    - `.recon` : Reconstruction techniques
    - `.sim` : Simulation functions
  - `.utils` : Utilities


## Features
- Full support for complex numbers
  - Adjoint takes the conjugate transpose
- Linear Operator Types
  - Diagonal: Elementwise multiply
  - Dense: Regular matrix multiplication
  - FFT: Fourier Transform
  - NUFFT: Non Uniform Fourier Transform
- Named Dimensions
  - Name the input and output dimensions of the linear operators
    - Helps with batching
  - TODO Graph the computation with names
- Slicing Linear Operators
  - Related to Split - Allows for batching the computation in arbitrary ways
    - Saves GPU memory
- Functional interface
  - For when the linear operators need to change e.g. each batch of data (machine learning)
  - Compatible with splitting/slicing of linops as well.
- TODO: Automatic simplification of chains of linear operators
  - For example, Split followed by Stack simplifies to Identity
    - ...if the splitting and stacking is done along the same dimension.

## Implementation Strategy
- Data structures
  - The linop data
    - Stores the actual data for the linop
    - Basically just a dictionary
  - The linop ordering
  - The linop execution strategy (schedule)
    - How the computation is actually batched and executed
      - Include the batching strategy for each input/output dim

## Batching strategy
- Batch object in tiling.py

# MRI

# Project organization

# Unit Testing

