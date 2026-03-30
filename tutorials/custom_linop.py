import marimo as mo

app = mo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Creating Custom Linear Operators

    This tutorial shows how to create your own `NamedLinop` subclasses.
    Every operator in torchlinops follows the same pattern, so once you
    understand the interface you can wrap any linear operation you need.

    ## The NamedLinop Interface

    To define a custom operator you need:

    1. **`__init__`** — call `super().__init__(NamedShape(ishape, oshape))`
       to declare the named input and output dimensions.
    2. **`fn(linop, x, /)`** — a `@staticmethod` that computes the forward
       operation $y = A x$.
    3. **`adj_fn(linop, x, /)`** — a `@staticmethod` that computes the
       adjoint operation $y = A^H x$.

    Optionally you can also override `normal_fn` (for an efficient $A^H A$),
    `split_forward` (for multi-GPU tiling), and `size` (to report dimension
    sizes).
    """)
    return


@app.cell
def _(mo):
    mo.md("## Setup")
    return


@app.cell
def _():
    import torch
    from torch import Tensor

    from torchlinops import Dense, Dim, NamedLinop
    from torchlinops.nameddim import NamedShape as NS
    from torchlinops.utils import is_adjoint

    torch.manual_seed(0)
    return Dense, Dim, NS, NamedLinop, Tensor, is_adjoint, torch


@app.cell
def _(mo):
    mo.md("""
    ## Example 1: Diagonal Scaling

    The simplest useful operator multiplies each element by a weight vector.
    This is mathematically $y = w \\odot x$ (elementwise product).

    We store the weight as an `nn.Parameter` so that it moves with the
    module when you call `.to(device)`. The `fn` and `adj_fn` static
    methods receive the linop instance as their first argument — this is how
    they access `self.weight` without being regular methods.
    """)
    return


@app.cell
def _(NS, NamedLinop, Tensor, torch):
    class DiagScale(NamedLinop):
        """Elementwise scaling: y = w * x."""

        def __init__(self, weight: Tensor, ioshape):
            # For a diagonal operator the input and output shapes are the same,
            # so we pass ioshape for both.
            super().__init__(NS(ioshape, ioshape))
            import torch.nn as nn

            self.weight = nn.Parameter(weight, requires_grad=False)

        @staticmethod
        def fn(linop, x, /):
            return x * linop.weight

        @staticmethod
        def adj_fn(linop, x, /):
            # The adjoint of elementwise multiplication by w is multiplication
            # by conj(w).
            return x * torch.conj(linop.weight)

        @staticmethod
        def normal_fn(linop, x, /):
            # A^H A x = |w|^2 * x — avoids two separate passes.
            return x * torch.abs(linop.weight) ** 2

    return (DiagScale,)


@app.cell
def _(mo):
    mo.md("Let's create an instance and try it out.")
    return


@app.cell
def _(DiagScale, Dim, torch):
    N = 8
    w = torch.randn(N, dtype=torch.complex64)
    D = DiagScale(w, ioshape=Dim("N"))

    x = torch.randn(N, dtype=torch.complex64)
    y = D(x)
    print("D(x) =", y)
    print("D.H(y) =", D.H(y))
    return D, N


@app.cell
def _(mo):
    mo.md("""
    ## Testing the Adjoint

    A correct adjoint must satisfy the identity
    $\\langle y, A x \\rangle = \\langle A^H y, x \\rangle$ for all $x, y$.
    The helper `is_adjoint` checks this numerically.
    """)
    return


@app.cell
def _(D, N, is_adjoint, torch):
    x_test = torch.randn(N, dtype=torch.complex64)
    y_test = torch.randn(N, dtype=torch.complex64)
    passed = is_adjoint(D, x_test, y_test)
    print(f"Adjoint test passed: {passed}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Example 2: Permutation Operator

    A permutation operator reorders the elements of a vector according to a
    fixed index mapping. Its adjoint is the *inverse* permutation (which is
    also its transpose, since permutation matrices are orthogonal).
    """)
    return


@app.cell
def _(NS, NamedLinop, Tensor, torch):
    class Permute(NamedLinop):
        """Reorder elements: y[i] = x[perm[i]]."""

        def __init__(self, perm: Tensor, ishape, oshape):
            super().__init__(NS(ishape, oshape))
            import torch.nn as nn

            # Store perm and its inverse as buffers so they travel with the module.
            self.perm = nn.Parameter(perm, requires_grad=False)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(len(perm))
            self.inv_perm = nn.Parameter(inv, requires_grad=False)

        @staticmethod
        def fn(linop, x, /):
            return x[linop.perm]

        @staticmethod
        def adj_fn(linop, x, /):
            # The adjoint of a permutation is the inverse permutation.
            return x[linop.inv_perm]

        def size(self, dim):
            if dim in self.ishape:
                return len(self.perm)
            if dim in self.oshape:
                return len(self.perm)
            return None

    return (Permute,)


@app.cell
def _(mo):
    mo.md("Create a random permutation and verify it.")
    return


@app.cell
def _(Dim, Permute, torch):
    M = 6
    perm = torch.randperm(M)
    P = Permute(perm, ishape=Dim("X"), oshape=Dim("Y"))

    x_perm = torch.arange(M, dtype=torch.float32)
    print("x       =", x_perm)
    print("P(x)    =", P(x_perm))
    print("P.H(P(x)) =", P.H(P(x_perm)))  # should recover x
    return M, P


@app.cell
def _(M, P, is_adjoint, torch):
    # The adjoint test should pass for the permutation operator too.
    x_test2 = torch.randn(M)
    y_test2 = torch.randn(M)
    passed2 = is_adjoint(P, x_test2, y_test2)
    print(f"Permutation adjoint test passed: {passed2}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Composing Custom and Built-in Operators

    One of the main benefits of the `NamedLinop` system is easy composition
    via the `@` operator. Here we compose our `DiagScale` with a built-in
    `Dense` matrix operator to form $A = D M$ where $D$ is diagonal
    scaling and $M$ is a dense matrix.
    """)
    return


@app.cell
def _(Dense, DiagScale, Dim, N, torch):
    K = 5
    M_weight = torch.randn(K, N, dtype=torch.complex64)
    M_op = Dense(
        weight=M_weight,
        weightshape=Dim("KN"),
        ishape=Dim("N"),
        oshape=Dim("K"),
    )

    # Compose: first apply M, then apply D in the K-space
    D_k = DiagScale(
        torch.randn(K, dtype=torch.complex64),
        ioshape=Dim("K"),
    )
    A = D_k @ M_op
    print("Composed operator:", A)
    return A, K


@app.cell
def _(mo):
    mo.md("""
    The composed operator automatically supports adjoint and normal
    operations.
    """)
    return


@app.cell
def _(A, K, N, is_adjoint, torch):
    x_comp = torch.randn(N, dtype=torch.complex64)
    y_comp = A(x_comp)
    x_adj = A.H(y_comp)
    print(f"Forward:  {x_comp.shape} -> {y_comp.shape}")
    print(f"Adjoint:  {y_comp.shape} -> {x_adj.shape}")

    x_t = torch.randn(N, dtype=torch.complex64)
    y_t = torch.randn(K, dtype=torch.complex64)
    print(f"Composed adjoint test passed: {is_adjoint(A, x_t, y_t)}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    To create a custom `NamedLinop`:

    1. Subclass `NamedLinop` and call
       `super().__init__(NS(ishape, oshape))` in `__init__`.
    2. Define `fn` and `adj_fn` as `@staticmethod` methods with
       signature `(linop, x, /)`.
    3. Optionally define `normal_fn` for an efficient $A^H A$.
    4. Use `is_adjoint` to verify correctness.
    5. Compose freely with other operators using `@`.
    """)
    return


if __name__ == "__main__":
    app.run()
