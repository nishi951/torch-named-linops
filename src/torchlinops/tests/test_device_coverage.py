import warnings

import torch

from torchlinops.linops.device import DeviceSpec, ToDevice


def test_device_spec_str():
    ds = DeviceSpec("cpu")
    assert ds.device == torch.device("cpu")
    assert ds.type == "cpu"


def test_todevice_cpu():
    td = ToDevice(torch.device("cpu"), torch.device("cpu"), ioshape=("N",))
    x = torch.randn(5)
    y = td(x)
    assert y.shape == x.shape


def test_todevice_adjoint():
    td = ToDevice(torch.device("cpu"), torch.device("cpu"))
    adj = td.adjoint()
    assert adj is not None


def test_todevice_normal():
    from torchlinops import Identity

    td = ToDevice(torch.device("cpu"), torch.device("cpu"))
    N = td.normal()
    assert isinstance(N, Identity)


def test_todevice_split():
    td = ToDevice(torch.device("cpu"), torch.device("cpu"))
    s = td.split_forward(None, None)
    assert s is not None


def test_todevice_repr():
    td = ToDevice(torch.device("cpu"), torch.device("cpu"))
    r = repr(td)
    assert "cpu" in r


def test_repeated_event_init():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from torchlinops.utils._event import RepeatedEvent

        ev = RepeatedEvent()
        assert ev.last_event is None
        assert "RepeatedEvent" in repr(ev)
