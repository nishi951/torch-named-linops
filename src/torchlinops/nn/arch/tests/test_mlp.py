import pytest

import torch

import torchlinops.nn as tlnn


def test_mlp():
    """Basic initialization test"""
    mlp = tlnn.MLP(
        in_chans=4,
        out_chans=7,
        hidden_chans=256,
        num_layers=3,
    )
    x = torch.randn(1, 4)
    y = mlp(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 7
    assert True
