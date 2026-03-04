"""
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
"""

# %%
# Setup
# -----

import torch

from torchlinops import Dense, Diagonal, Dim
from torchlinops.alg import conjugate_gradients, power_method
from torchlinops.utils import is_adjoint

torch.manual_seed(0)

# %%
# Creating a Dense Operator
# -------------------------
# The simplest operator in torchlinops is ``Dense``, which wraps a matrix
# and gives each axis a name.  The ``Dim()`` helper turns a compact string
# into a tuple of dimension names by splitting on uppercase letters:

print("Dim('MN')  =", Dim("MN"))  # ('M', 'N')
print("Dim('NxNy') =", Dim("NxNy"))  # ('Nx', 'Ny')

# You can also write dimension names as a plain tuple:
print("('M', 'N') works too:", ("M", "N"))

# %%
# Let's create a 4×3 matrix and wrap it as a Dense operator whose
# input dimension is ``N`` (size 3) and output dimension is ``M`` (size 4).

W = torch.randn(4, 3)
A = Dense(W, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
print(A)

# %%
# Applying the Operator
# ---------------------
# A named linop is callable just like a function.  Calling ``A(x)`` applies
# the forward operation — in this case, a matrix-vector multiply.

x = torch.randn(3)
y = A(x)
print("A(x) shape:", y.shape)

# We can verify this matches a plain matrix-vector multiply:
y_manual = W @ x
print("W @ x matches A(x):", torch.allclose(y, y_manual))

# %%
# The Diagonal Operator
# ---------------------
# ``Diagonal`` represents element-wise multiplication by a weight vector:
# $D(x) = w \odot x$.  Input and output shapes are the same.

w = torch.tensor([1.0, 2.0, 3.0])
D = Diagonal(w, ioshape=Dim("N"))
print(D)

x = torch.randn(3)
print("D(x):", D(x))
print("w * x:", w * x)
print("Match:", torch.allclose(D(x), w * x))

# %%
# Composing Operators with @
# --------------------------
# Named linops compose with ``@``, just like matrix multiplication.
# Composing two operators creates a ``Chain`` that applies them right-to-left:
# ``(B @ A)(x)`` means ``B(A(x))``.

W1 = torch.randn(4, 3)
W2 = torch.randn(5, 4)

A = Dense(W1, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
B = Dense(W2, weightshape=Dim("KM"), ishape=Dim("M"), oshape=Dim("K"))

C = B @ A  # Chain: first A, then B
print("Composed operator:", C)

x = torch.randn(3)
y_chain = C(x)
y_manual = W2 @ (W1 @ x)
print("(B @ A)(x) matches W2 @ W1 @ x:", torch.allclose(y_chain, y_manual, atol=1e-6))

# %%
# Adjoints with .H
# ----------------
# Every named linop has an adjoint, accessed via ``.H``.
# For a Dense operator wrapping matrix $W$, the adjoint applies $W^H$
# (conjugate transpose).

A = Dense(W1, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
print("A  : ishape =", A.ishape, " oshape =", A.oshape)
print("A.H: ishape =", A.H.ishape, " oshape =", A.H.oshape)

# Verify: A.H(y) should equal W^T @ y (for real matrices)
y = torch.randn(4)
print("A.H(y) matches W.T @ y:", torch.allclose(A.H(y), W1.T @ y))

# %%
# We can verify the adjoint relationship numerically using the
# ``is_adjoint`` utility:  <y, A x> == <A^H y, x>

x = torch.randn(3)
y = torch.randn(4)
print("Adjoint test passed:", is_adjoint(A, x, y).item())

# %%
# The Normal Operator .N
# ----------------------
# The **normal operator** is $A^N = A^H A$.  It is always square and
# positive semi-definite, making it suitable for iterative solvers like
# conjugate gradients.

A = Dense(W1, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
AN = A.N
print("A.N:", AN)
print("A.N ishape:", AN.ishape, " oshape:", AN.oshape)

x = torch.randn(3)
print("A.N(x) matches A.H(A(x)):", torch.allclose(AN(x), A.H(A(x)), atol=1e-6))

# %%
# Solving a Linear System
# -----------------------
# A common task is solving $Ax = b$ in the least-squares sense.  We form the
# **normal equation** $A^H A x = A^H b$ and solve it with conjugate gradients.
#
# Let's set up a small system and recover the solution.

# Create a well-conditioned operator
M, N = 8, 5
W = torch.randn(M, N)
A = Dense(W, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))

# Generate ground-truth and observed data
x_true = torch.randn(N)
b = A(x_true)

# %%
# First, estimate the largest eigenvalue of $A^H A$ with the power method
# to gauge the condition of the system:

_, eigval = power_method(
    A.N, torch.randn(N), max_iters=30, tqdm_kwargs=dict(leave=False)
)
print(f"Largest eigenvalue of A.N: {eigval.item():.4f}")

# %%
# Now solve $A^H A x = A^H b$ using conjugate gradients:

rhs = A.H(b)
x_cg = conjugate_gradients(A.N, rhs, max_num_iters=50, gtol=1e-6)

print(f"x_true: {x_true}")
print(f"x_cg:   {x_cg}")
print(f"Relative error: {torch.norm(x_cg - x_true) / torch.norm(x_true):.2e}")

# %%
# Summary
# -------
# In this tutorial you learned the building blocks of ``torchlinops``:
#
# - **Dim()** — compact named dimension strings
# - **Dense** — matrix-vector multiply with named axes
# - **Diagonal** — element-wise scaling
# - **@ composition** — chaining operators into pipelines
# - **.H** — the adjoint operator
# - **.N** — the normal operator ($A^H A$)
# - **power_method** and **conjugate_gradients** — iterative algorithms
#
# These primitives compose to build complex linear systems
# while keeping dimension bookkeeping clear and automatic.
