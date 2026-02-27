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


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 GPUs are required for this test",
)
def test_todevice_streams():
    idevice = torch.device("cuda:0")
    odevice = torch.device("cuda:1")

    D2D2 = ToDevice(idevice, odevice)

    x = torch.randn(3, 4, device=idevice)

    # Test with first device
    y1 = D2D2(x)
    assert y1.device == odevice

    # Test with second device
    z1 = D2D2.H(y1)
    assert z1.device == idevice

    w2 = D2D2.N(x)
    assert w2.device == x.device

    # Verify that the streams are used
    assert D2D2.ispec.transfer_stream is not None
    assert D2D2.ospec.compute_stream is not None

    print(D2D2)
