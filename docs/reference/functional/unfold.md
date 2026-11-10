# Unfold / Fold

Differentiable block extraction (unfold) and block reassembly (fold) operations,
supporting 1D/2D/3D with optional masking and Triton backends.

`array_to_blocks` and `blocks_to_array` are the supported public API. The raw
`unfold` / `fold` kernels are internal implementation details; use the wrappers
when you need autograd support.

::: torchlinops.functional.array_to_blocks

::: torchlinops.functional.blocks_to_array
