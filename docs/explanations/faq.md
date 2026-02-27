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

## When should I use `.H` (adjoint) vs `.N` (normal)?

The adjoint and normal operators serve different purposes:

### When to use `.H` (adjoint)

- **Gradient backpropagation**: In optimization or deep learning, the gradient through a linear operator $A$ involves $A^H$. For example, if $y = Ax$ and you're differentiating a loss $L(y)$, then $\frac{\partial L}{\partial x} = A^H \frac{\partial L}{\partial y}$.
- **Computing inner products**: The adjoint lets you compute $\langle Ax, y \rangle$ as $\langle x, A^H y \rangle$, which is useful when one side is cheaper to compute.
- **Solving adjoint equations**: In inverse problems, you often need to solve $A^H A x = A^H b$.

### When to use `.N` (normal)

- **Conjugate gradient solves**: CG solves $A x = b$ by working with $A^H A$ (the normal equations). Use `A.N` directly with `conjugate_gradients(A.N, A.H(b))`.
- **Preconditioning**: Normal operators are always square and (semi)definite, making them suitable for building preconditioners.
- **Eigenvalue problems**: The normal $A^H A$ has real, non-negative eigenvalues. Use with `power_method(A.N, ...)` to find the largest eigenvalue.

In short: use `.H` when you need the "backward" direction of the operator, and use `.N` when you need a symmetric, positive-semidefinite operator for iterative solvers.

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

## Troubleshooting

### Shape mismatch errors when composing linops

If you get a shape mismatch error when using `@` to compose linops, the output shape of the first operator must match the input shape of the next:

```python
# Error: A has oshape (M,) but B has ishape (N,)
# A @ B  # Shape mismatch!

# Fix: Check that dimension names match
A = Dense(W, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
B = Diagonal(d, ioshape=Dim("M"))  # Same dimension name as A's oshape
C = B @ A  # Works! (M,) -> (M,)
```

Use `print(A.ishape, A.oshape)` to inspect operator shapes and find the mismatch.

### Adjoint test fails (`is_adjoint` returns False)

If the adjoint test fails, common causes include:

1. **Missing conjugate for complex numbers**: For complex inputs, the adjoint must include the conjugate transpose, not just the transpose.
2. **Sign errors**: Check that your `adj_fn` is the exact mathematical adjoint, not just similar.
3. **Normalization**: Make sure FFT and other operators use the correct normalization (orthonormal by default).

```python
# Example: verifying a custom adjoint
from torchlinops.utils import is_adjoint
A = MyLinop(...)
x = torch.randn(8, dtype=torch.complex64)
y = torch.randn(8, dtype=torch.complex64)
print(is_adjoint(A, x, y))  # Should be True for correct adjoint
```

### CUDA out of memory when creating large linops

Large linops (especially interpolation tables, FFT plans, or dense matrices) can exhaust GPU memory. Solutions:

1. **Create on CPU first, then move to GPU**:
   ```python
   linop = Dense(weight_cpu, ...)
   linop = linop.to("cuda")
   ```

2. **Use float32 instead of float64** if precision allows.

3. **Lazy initialization**: For very large operators, consider whether you need all the data in memory at once.

### Linop doesn't support gradient backpropagation

Linops store weights as `nn.Parameter` with `requires_grad=False` by default (since linear operators are typically fixed in optimization problems). To make a linop trainable:

```python
# Weights are trainable if you set requires_grad=True
weight = nn.Parameter(torch.randn(M, N), requires_grad=True)
A = Dense(weight, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))

# Now A(x) will support autograd
loss = (A(x) ** 2).sum()
loss.backward()
print(weight.grad)  # Gradient d(loss)/d(weight)
```

## Performance Tips

### Chain overhead is minimal

Composing linops with `@` creates a `Chain` that adds minimal overhead—the actual computation is still just the individual operator forward passes. Don't avoid composition for performance reasons; the abstraction cost is negligible compared to the actual math.

### GPU memory considerations

- **Interpolation tables**: The `Interpolate` operator uses Kaiser-Bessel or spline kernels stored in memory. For large grids, these tables can be large. Consider the trade-off between kernel accuracy and memory.
- **FFT plans**: FFT operations may pre-compute plans. These are typically small but can add up with many different sizes.
- **Dense matrices**: A dense matrix of size $10000 \times 10000$ uses ~800MB in float32. Consider whether a different operator (sparse, diagonal, FFT-based) could work.

### When to use the functional interface directly

The functional interface (`torchlinops.functional`) provides raw tensor operations without the linop abstraction. Use it when:

1. You're in a tight inner loop where linop call overhead matters
2. You only need the forward pass, not adjoint/normal
3. You're building a custom linop and need full control

For most users and use cases, the linop interface is preferred for its clarity and automatic adjoint/normal support.
