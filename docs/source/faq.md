# FAQ

## Why torch?
Torch is a mature, actively maintained, and performant framework for GPU
computing. However, we note that the backend is not tied fundamentally to
pytorch (or any autodiff framework) and could in theory be swapped for any
suitable numerical computing framework.

## Why "names"?
In many numerical computing applications, each dimension of a multidimensional
arrays has an associated "label". For example, a grayscale image may have a height axis
`H` and a width axis `W`; introducing color gives a color axis `C`, and so on.
Operations on this array are often sensitive to the ordering of these axes. For
example, conversion from RGB to grayscale is a transformation with "shape" `CHW -> HW`.

Other examples of transformations are:
- Expansion (or "unsqueezing"): `ABC -> A1BC`
- Reduction: `XYZ -> YZ`
- Fourier transform: `XY -> KxKy`
- Batched matrix multiply: `BMN -> BMP`
- ...and many more.

By including the shape specifications in our linear operators, it becomes
possible to write self-documenting code - code that does not require extensive
comments or docstrings to elucidate its meaning. This idea was partially inspired by
[einops](https://einops.rocks), a powerful tool for tensor shape manipulation
that uses a simple and clear syntax to solve an annoying problem.


