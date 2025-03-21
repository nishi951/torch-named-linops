import pytest

import torch
from torchlinops import ToDevice


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_todevice():
    idevice = torch.device("cpu")
    odevice = torch.device("cuda:0")
    D2D = ToDevice(idevice, odevice)
    x = torch.randn(3, 4)
    y = D2D(x)
    assert y.device == odevice

    z = D2D.H(y)
    assert z.device == idevice

    w = D2D.N(x)
    assert w.device == x.device
    print(D2D)
