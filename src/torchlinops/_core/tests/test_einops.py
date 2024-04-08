import torch

from torchlinops import SumReduce, Repeat

from adjoint_helpers import is_adjoint


def test_reduce_repeat():
    M, N = 5, 7
    x = torch.randn(M, N)
    y = torch.randn(N)

    A = SumReduce(("M", "N"), ("N",))
    B = Repeat({"M": M}, ("N",), ("M", "N"))

    assert is_adjoint(A, x, y)
    assert is_adjoint(B, y, x)
