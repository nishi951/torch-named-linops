import pytest

import torch

from torchlinops import Diagonal, Batch, Dense
from torchlinops.mri import SENSE
from torchlinops.utils import same_storage


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


def test_split():
    mps = torch.randn(4, 8, 8, 8, dtype=torch.complex64)
    S = SENSE(mps)
    ibatch = [slice(None), slice(None), slice(None)]
    obatch = [slice(0, 1), slice(None), slice(None), slice(None)]
    S_split = S.split(S, ibatch, obatch)

    assert tuple(S_split.mps.shape) == (1, 8, 8, 8)
    assert tuple(S.mps.shape) == (4, 8, 8, 8)
    assert (S.mps[0] == S_split.mps[0]).all()
    assert same_storage(S.mps, S_split.mps)
    # from pytorch_memlab import MemReporter
    # MemReporter().report()

def test_sense_batch():
    mps = torch.randn(4, 8, 8, 8, dtype=torch.complex64)
    D = Dense(torch.randn(4, 4), weightshape=("C", "C1"),
              ishape=('C', 'Nx', 'Ny', 'Nz'),
              oshape=('C1', 'Nx', 'Ny', 'Nz'))

    S = SENSE(mps)
    A = D @ S
    A_batch = Batch(
        A,
        input_device='cpu',
        output_device='cpu',
        input_dtype=torch.complex64,
        output_dtype=torch.complex64,
        C=1,
    )
    #print(A_batch.N)
    print(A_batch.H)
    print(A.N)
    print(A.H)

def test_sense_adjoint_methods():
    mps = torch.randn(4, 8, 8, 8, dtype=torch.complex64)
    S = SENSE(mps)
    SH = S.H

    # S and SH form a pair
    assert id(S.H) == id(SH)
    assert id(SH.H) == id(S)
