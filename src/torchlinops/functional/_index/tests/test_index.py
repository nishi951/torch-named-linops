import pytest

import torch
from torchlinops.functional import index, index_adjoint


def test_index_2d_bcast():
    idx = (torch.tensor([1, 0, 1])[:, None], torch.tensor([1, 2, 3])[None, :])
    vals = torch.arange(8).reshape(2, 4)
    out = vals[idx]

    out2 = index(vals, idx)
    assert (out == out2).all()


def test_index_2d_bcast_slicing():
    tidx = (torch.tensor([0, 1])[:, None], torch.tensor([1, 2, 3])[None, :])
    vals = torch.arange(8).reshape(2, 4)
    out = vals[tidx]

    idx = (slice(None), slice(1, 4))
    out2 = index(vals, idx)
    assert (out == out2).all()


def test_index_adjoint():
    idx = torch.tensor([1, 3, 2])
    idx = (idx,)
    vals = torch.tensor([[5.0, 4.0, -1.0], [5.0, 4.0, -1.0]])
    oshape = (4,)
    out = index_adjoint(vals, idx, oshape)
    assert (out[0] == torch.tensor([0.0, 5.0, -1.0, 4.0])).all()
    assert (out[0] == out[1]).all()
