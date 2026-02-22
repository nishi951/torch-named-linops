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

## How do I create a custom linop?

Subclass `NamedLinop` and implement:

1. **`__init__`**: Call `super().__init__(NS(ishape, oshape))` to set the shape.
2. **`fn` (staticmethod)**: The forward pass, `fn(linop, x) -> y`.
3. **`adj_fn` (staticmethod)**: The adjoint pass, `adj_fn(linop, x) -> y`.
4. **`split_forward`** (optional): How to split along named dimensions.

See the [Custom Linops](../howto/custom_linops.md) guide for the full details and conventions.

## What does `.H` do internally?

Accessing `A.H` creates a shallow copy of `A` with:

- The shape flipped: `ishape` and `oshape` are swapped.
- The functions swapped: `fn` and `adj_fn` are exchanged.

The result is cached, so `A.H` always returns the same object. And since the adjoint links back to the original (`A.H._adjoint = [A]`), calling `A.H.H` returns `A` without creating a new copy.

Because the copy is shallow, the adjoint shares the same weight tensors as the original. Updating weights on one is immediately reflected in the other.

## What does `.N` do internally?

Accessing `A.N` creates a linop representing the normal operator $A^H A$. By default, this calls `adj_fn(fn(x))` -- the naive compose-and-apply approach.

Many linops override `normal()` with optimized implementations:

- **FFT**: $F^H F = I$ (returns `Identity`).
- **Diagonal**: $(D^H D)(x) = |w|^2 \odot x$ (single elementwise multiply).
- **Chain**: Folds `normal()` through the chain, enabling Toeplitz embedding and other optimizations.

Like `.H`, the result is cached after first access.

## What are `...` and `()` in shapes?

These are special wildcard dimensions used for flexible shape matching:

- **`...` (Ellipses)**: Matches zero or more dimensions of any name. Analogous to Python's `Ellipsis`. For example, `("...", "K")` matches `("Batch", "Coil", "K")`.
- **`()` (Any)**: Matches exactly one dimension of any name. For example, `("()", "K")` matches `("Batch", "K")` but not `("Batch", "Coil", "K")`.

These are primarily used internally for shape compatibility checks during composition and dimension updates.

## How does multi-GPU splitting work?

The library can decompose a linop into tiles along its named dimensions and place each tile on a different GPU. The key components are:

1. **`BatchSpec`**: Defines chunk sizes and device placement.
2. **`create_batched_linop`**: Orchestrates the split, placement, and reassembly.
3. **`ToDevice`**: Handles data transfer between devices using CUDA streams.

The result is a composite linop (tree of `Concat`/`Add` operators) that behaves identically to the original but executes across multiple devices. See [Multi-GPU Splitting](multi_gpu.md) for the full explanation.
