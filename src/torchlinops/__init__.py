"""torch-named-linops: A flexible linear operator abstraction for PyTorch.

This library provides named linear operators for matrix-free numerical computing,
optimization, and machine learning. Operators use named dimensions to make
composition, adjoint creation, and multi-GPU splitting simple and unambiguous.

Key classes:
    NamedLinop: Base class for all named linear operators.
    Dense, Diagonal, FFT, NUFFT: Core operator implementations.
    Chain, Add: Composition operators.
    Dim: Convenience constructor for named dimensions.

Example:
    >>> import torch
    >>> from torchlinops import Dense, Dim
    >>> w = torch.randn(3, 7)
    >>> A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
    >>> x = torch.randn(7)
    >>> y = A(x)  # Apply the operator
    >>> z = A.H(y)  # Apply the adjoint
"""

from .alg import *
from .linops import *
from .nameddim import *

__version__ = "0.6.0"
