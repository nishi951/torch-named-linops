# Interpolation

Differentiable interpolation (ungridding) and adjoint interpolation (gridding) functions.

`interpolate` and `interpolate_adjoint` are the supported public API. The raw
`grid` / `ungrid` kernels are internal implementation details; use the wrappers
when you need autograd support.

::: torchlinops.functional.interpolate

::: torchlinops.functional.interpolate_adjoint
