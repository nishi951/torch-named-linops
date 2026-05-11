import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""
    # Getting Started with Named Linear Operators

    This tutorial introduces the core concepts of `torchlinops` — a library for
    building and composing **named linear operators** in PyTorch.

    You will learn how to:

    - Create operators with named dimensions using `Dim()`
    - Apply operators and verify correctness
    - Use the `Diagonal` operator
    - Compose operators with `@`
    - Compute adjoints (`.H`) and normal operators (`.N`)
    - Solve a linear system with `conjugate_gradients`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Setup
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch

    from torchlinops import Dense, Diagonal, Dim
    from torchlinops.alg import conjugate_gradients, power_method
    from torchlinops.utils import is_adjoint

    torch.manual_seed(0)
    return (
        Dense,
        Diagonal,
        Dim,
        conjugate_gradients,
        is_adjoint,
        mo,
        power_method,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Creating a Dense Operator

    The simplest operator in torchlinops is `Dense`, which wraps a matrix
    and gives each axis a name. The `Dim()` helper turns a compact string
    into a tuple of dimension names by splitting on uppercase letters:
    """)
    return


@app.cell
def _(Dim):
    print("Dim('MN')  =", Dim("MN"))  # ('M', 'N')
    print("Dim('NxNy') =", Dim("NxNy"))  # ('Nx', 'Ny')

    # You can also write dimension names as a plain tuple:
    print("('M', 'N') works too:", ("M", "N"))
    return


@app.cell
def _(mo):
    mo.md("""
    Let's create a 4×3 matrix and wrap it as a Dense operator whose
    input dimension is `N` (size 3) and output dimension is `M` (size 4).
    """)
    return


@app.cell
def _(Dense, Dim, torch):
    W = torch.randn(4, 3)
    A = Dense(W, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
    print(A)
    return A, W


@app.cell
def _(mo):
    mo.md("""
    ## Applying the Operator

    A named linop is callable just like a function. Calling `A(x)` applies
    the forward operation — in this case, a matrix-vector multiply.
    """)
    return


@app.cell
def _(A, W, torch):
    _x = torch.randn(3)
    _y = A(_x)
    print("A(x) shape:", _y.shape)
    _y_manual = W @ _x
    # We can verify this matches a plain matrix-vector multiply:
    print("W @ x matches A(x):", torch.allclose(_y, _y_manual))
    return


@app.cell
def _(mo):
    mo.md("""
    ## The Diagonal Operator

    `Diagonal` represents element-wise multiplication by a weight vector:
    $D(x) = w \odot x$. Input and output shapes are the same.
    """)
    return


@app.cell
def _(Diagonal, Dim, torch):
    w = torch.tensor([1.0, 2.0, 3.0])
    D = Diagonal(w, ioshape=Dim("N"))
    print(D)
    _x = torch.randn(3)
    print("D(x):", D(_x))
    print("w * x:", w * _x)
    print("Match:", torch.allclose(D(_x), w * _x))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Composing Operators with @

    Named linops compose with `@`, just like matrix multiplication.
    Composing two operators creates a `Chain` that applies them right-to-left:
    `(B @ A)(x)` means `B(A(x))`.
    """)
    return


@app.cell
def _(Dense, Dim, torch):
    W1 = torch.randn(4, 3)
    W2 = torch.randn(5, 4)
    A_1 = Dense(W1, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
    B = Dense(W2, weightshape=Dim("KM"), ishape=Dim("M"), oshape=Dim("K"))
    C = B @ A_1
    print("Composed operator:", C)
    _x = torch.randn(3)
    y_chain = C(_x)
    _y_manual = W2 @ (W1 @ _x)
    print(
        "(B @ A)(x) matches W2 @ W1 @ x:",
        torch.allclose(y_chain, _y_manual, atol=1e-06),
    )
    return (W1,)


@app.cell
def _(mo):
    mo.md("""
    ## Adjoints with .H

    Every named linop has an adjoint, accessed via `.H`.
    For a Dense operator wrapping matrix $W$, the adjoint applies $W^H$
    (conjugate transpose).
    """)
    return


@app.cell
def _(Dense, Dim, W1, torch):
    A_2 = Dense(W1, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
    print("A  : ishape =", A_2.ishape, " oshape =", A_2.oshape)
    print("A.H: ishape =", A_2.H.ishape, " oshape =", A_2.H.oshape)
    _y = torch.randn(4)
    print("A.H(y) matches W.T @ y:", torch.allclose(A_2.H(_y), W1.T @ _y))
    return (A_2,)


@app.cell
def _(mo):
    mo.md("""
    We can verify the adjoint relationship numerically using the
    `is_adjoint` utility: $\langle y, A x \rangle = \langle A^H y, x \rangle$
    """)
    return


@app.cell
def _(A_2, is_adjoint, torch):
    _x = torch.randn(3)
    _y = torch.randn(4)
    print("Adjoint test passed:", is_adjoint(A_2, _x, _y).item())
    return


@app.cell
def _(mo):
    mo.md("""
    ## The Normal Operator .N

    The **normal operator** is $A^N = A^H A$. It is always square and
    positive semi-definite, making it suitable for iterative solvers like
    conjugate gradients.
    """)
    return


@app.cell
def _(Dense, Dim, W1, torch):
    A_3 = Dense(W1, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
    AN = A_3.N
    print("A.N:", AN)
    print("A.N ishape:", AN.ishape, " oshape:", AN.oshape)
    _x = torch.randn(3)
    print(
        "A.N(x) matches A.H(A(x)):", torch.allclose(AN(_x), A_3.H(A_3(_x)), atol=1e-06)
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Solving a Linear System

    A common task is solving $Ax = b$ in the least-squares sense. We form the
    **normal equation** $A^H A x = A^H b$ and solve it with conjugate gradients.

    Let's set up a small system and recover the solution.
    """)
    return


@app.cell
def _(Dense, Dim, torch):
    M, N = (8, 5)
    # Create a well-conditioned operator
    W_1 = torch.randn(M, N)
    A_4 = Dense(W_1, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
    x_true = torch.randn(N)
    # Generate ground-truth and observed data
    b = A_4(x_true)
    return A_4, N, b, x_true


@app.cell
def _(mo):
    mo.md("""
    First, estimate the largest eigenvalue of $A^H A$ with the power method
    to gauge the condition of the system:
    """)
    return


@app.cell
def _(A_4, N, power_method, torch):
    _, eigval = power_method(
        A_4.N, torch.randn(N), max_iters=30, tqdm_kwargs=dict(leave=False)
    )
    print(f"Largest eigenvalue of A.N: {eigval.item():.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    Now solve $A^H A x = A^H b$ using conjugate gradients:
    """)
    return


@app.cell
def _(A_4, b, conjugate_gradients, torch, x_true):
    rhs = A_4.H(b)
    x_cg = conjugate_gradients(A_4.N, rhs, max_num_iters=50, gtol=1e-06)
    print(f"x_true: {x_true}")
    print(f"x_cg:   {x_cg}")
    print(f"Relative error: {torch.norm(x_cg - x_true) / torch.norm(x_true):.2e}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    In this tutorial you learned the building blocks of `torchlinops`:

    - **Dim()** — compact named dimension strings
    - **Dense** — matrix-vector multiply with named axes
    - **Diagonal** — element-wise scaling
    - **@ composition** — chaining operators into pipelines
    - **.H** — the adjoint operator
    - **.N** — the normal operator ($A^H A$)
    - **power_method** and **conjugate_gradients** — iterative algorithms

    These primitives compose to build complex linear systems
    while keeping dimension bookkeeping clear and automatic.
    """)
    return


if __name__ == "__main__":
    app.run()
