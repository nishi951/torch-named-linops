# torch-named-linops
A flexible linear operator abstraction implemented in PyTorch.

Includes:

- Functionality for creating and manipulating linop dimensions.
- A useful set of core linops, including dense, diagonal, padding/cropping, and FFTs.
- Tools for composing or decomposing linops, including Chaining, Adding, and Batching
  (splitting) them.

- A set of mri-specific functionality including:
  - Special linops for sensitivity maps, density compensation, and non-uniform FFTs (NUFFTs)
  - Coil sensitivity map estimation
  - Functions for reconstruction such as conjugate gradient and FISTA, as well as
    implementations of certain priors.

Note: this project is Unrelated to the (also good) [torch_linops](https://github.com/cvxgrp/torch_linops)


## A note on monorepo-style git submodule dependencies
Many of my projects begin as isolated repos with self-contained pyproject.toml
files for ease of installation. However, this can make it inconvenient to
include these repos as git submodules in other repos, because they come with all
the "extra" python packaging files. These modules end up clogging up the source tree.


## Roadmap
Features
- [ ] Automatic shape determination, where possible (or at least some good defaults)
- [ ] Better error messages for linop chains and batches
- [ ] NUFFT backend checks for installed modules
- [ ] Batching over `Add`
- [ ] Unitary minus and `Subtract`
- [ ] `Stack` - combine linops across dimensions

Unit testing
- [ ] All linops adjoint tested
- [ ] All linops .to(device) tested
- [ ] Add code coverage

Infrastructure
- [ ] pre-commit with ruff
- [ ] Automatic dependency range management (dependabot?)
- [ ] Github actions including `build` and `twine`

Documentation
- [ ] MyST (Markdown intead of ReST)
- [ ] How-to guide for making new NamedLinops
- [ ] How-to guides for spiral SENSE, gridding recon, and maybe time segmentation
- [ ] Marimo notebook tutorials
- [ ] Autodoc - scraping docs from docstrings
- [ ] Explanation docs to catalog design decisions

PyPI (0.2 milestone)
Reducing dependencies
- [ ] (sigpy) NUFFT backend check for sigpy nufft backend
- [ ] (igrog) Self-contained fallback implicit GROG implementation?
- [ ] (optimized-mrf) Merge with or add as dependency to mr_sim repo
- [ ] (mr_sim) Remove entirely (simulation is not core functionality)
  - [ ] Save some of it for testing?
  - [ ] Add some simulation in `examples` directory?
- [ ] (mr_recon) Remove entirely (exists within igrog)

S-LORAKS
- [ ] S-LORAKS operator
- [ ] S-LORAKS proximal operator
- [ ] S-LORAKS majorization minimization operator

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

### Custom Linops
Custom linops should satisfy the following:
- `@staticmethod` `.fn()`, `.adj_fn()`, and `.normal_fn()`
  - Allows for swapping of functions rather than `bound_methods` in `adjoint()`.
- Calling the constructor of itself via `type(self)` inside `.split_forward()`.
  - This is because of how pytorch handles `copy`-ing parameters. We call the
    constructor rather than deepcopy to avoid making an extra copy of tensors.
    However, `copy` does not propery isolate the new parameter - modifications
    to the parameter will propagate up to the calling function.
- Use of `super().__init__(NS(ishape, oshape))` in the constructor.
  - This initializes the linop's shape
  - May change later to not requires `NS`

    



## Batching strategy
- Batch object in tiling.py
- Only applies to chains - don't batch over Adds
  - TODO: Figure out batching with Adds inside other Chains etc...

# MRI

# Project organization

# Unit Testing

# Notes
### Splitting Linops and copy/deepcopy
After some frustration, it turns out that the easiest way to slice a linop is to
`copy.deepcopy` it and then replace the copy's Parameters with split versions of the
original parameters. Basically, the trifecta of:
1. `nn.Module + nn.Parameter` for easy device management and tensor registration,
2. The need to copy non-tensor/non-parameter attributes exactly, without
   invoking the constructor, and
3. The need to reuse memory.
all make this a tricky problem to solve.

Because of this, I will probably go back to using the constructor (via
`type(self)(*args, **kwargs)``).
