import pytest

import torch

from torchlinops.mri import NUFFT
from torchlinops.utils import is_adjoint


@pytest.fixture(
    params=[
        None,
        {"plan_ahead": "gpu", "img_batch_size": 1},
    ]
)
def nufft(request):
    extras = request.param
    T, R, K = 3, 4, 5
    D = 2
    trj = 64 * (torch.rand((T, R, K, D)) - 0.5)
    F = NUFFT(
        trj,
        im_size=(64, 64),
        in_batch_shape=tuple(),
        out_batch_shape=("T", "R", "K"),
        toeplitz=False,
        backend="fi",
        extras=extras,
    )
    return F


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_nufft_planning(nufft):
    F = nufft

    if F.plan:
        device = F.plan_device
    else:
        device = torch.device("cpu")
    x = torch.randn(64, 64, dtype=torch.complex64, device=device)
    y = torch.randn(3, 4, 5, dtype=torch.complex64, device=device)

    if device != "cpu" and F.plan:
        assert is_adjoint(F, x, y, atol=1e-2, rtol=1e-2)
    else:
        assert is_adjoint(F, x, y)


@pytest.fixture(
    params=[
        {"plan_ahead": "gpu", "img_batch_size": 2},
    ]
)
def big_nufft(request):
    extras = request.param
    T, R, K = 500, 48, 1678
    D = 3
    trj = 64 * (torch.rand((T, R, K, D)) - 0.5)
    F = NUFFT(
        trj,
        im_size=(220, 220, 220),
        in_batch_shape=tuple(),
        out_batch_shape=("T", "R", "K"),
        toeplitz=False,
        backend="fi",
        extras=extras,
    )
    return F


@pytest.mark.gpu
@pytest.mark.big
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_big_nufft_planning(big_nufft):
    F = big_nufft

    if F.plan:
        device = F.plan_device
    else:
        device = torch.device("cpu")
    x = torch.randn(2, 220, 220, 220, dtype=torch.complex64, device=device)
    y = torch.randn(2, 500, 48, 1678, dtype=torch.complex64, device=device)

    breakpoint()
    if device != "cpu" and F.plan:
        assert is_adjoint(F, x, y, atol=1e-2, rtol=1e-2)
    else:
        assert is_adjoint(F, x, y)
