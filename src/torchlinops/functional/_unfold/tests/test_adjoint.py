import pytest
import torch

from torchlinops.functional import unfold
from torchlinops.functional import fold
from torchlinops.functional._unfold.nblocks import get_nblocks
from torchlinops.functional._unfold.array_to_blocks import get_norm_weights


PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
@pytest.mark.parametrize(
    "spec",
    [
        "small1d",
        "medium1d",
        pytest.param("large1d", marks=PYTEST_GPU_MARKS),
        "small2d",
        pytest.param("medium2d", marks=PYTEST_GPU_MARKS),
        pytest.param("large2d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_adjoint(dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32
    spec["nblocks"] = get_nblocks(spec["shape"], spec["block_size"], spec["stride"])

    ishape = (*spec["N"], *spec["shape"])
    oshape = (*spec["N"], *spec["nblocks"], *spec["block_size"])

    x = torch.randn(ishape, dtype=dtype, device=device)
    y = torch.randn(oshape, dtype=dtype, device=device)

    Ax = unfold(x, spec["block_size"], spec["stride"])
    AHy = fold(y, spec["shape"], spec["block_size"], spec["stride"])

    assert zdot(x, AHy).allclose(zdot(y, Ax).conj(), rtol=1e-4)


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
@pytest.mark.parametrize(
    "spec",
    [
        "small1d",
        "medium1d",
        pytest.param("large1d", marks=PYTEST_GPU_MARKS),
        "small2d",
        pytest.param("medium2d", marks=PYTEST_GPU_MARKS),
        pytest.param("large2d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_norm_weights(dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32
    spec["nblocks"] = get_nblocks(spec["shape"], spec["block_size"], spec["stride"])

    ishape = (*spec["N"], *spec["shape"])
    # oshape = (*spec["N"], *spec["nblocks"], *spec["block_size"])

    weights = get_norm_weights(
        spec["shape"], spec["block_size"], spec["stride"], device=device
    )
    weights = 1.0 / torch.clamp(weights, min=1e-6)
    x = torch.randn(ishape, dtype=dtype, device=device)
    x2 = weights * fold(
        unfold(x, spec["block_size"], spec["stride"]),
        spec["shape"],
        spec["block_size"],
        spec["stride"],
    )
    assert torch.allclose(x, x2)


def zdot(x, y):
    return torch.sum(x.conj() * y)
