import pytest

import torch

from torchlinops import Diagonal
from torchlinops.mri import SENSE


def test_normal():
    mps = torch.randn(4, 8, 8, 8, dtype=torch.complex64)
    diag = torch.randn_like(mps)
    x = torch.randn(8, 8, 8, dtype=torch.complex64)
    S = SENSE(mps)
    D = Diagonal(diag, S.oshape)

    SHS = S.N
    assert SHS.ishape == S.ishape
    assert torch.isclose(S.N(x), S.H(S(x))).all()

    DS = D @ S
    SHDDS = (D @ S).N
    assert SHDDS.oshape == S.ishape
    assert torch.isclose(DS.N(x), DS.H(DS(x))).all()
