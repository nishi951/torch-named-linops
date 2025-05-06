import pytest
from math import prod

import torch
import torchlinops.functional as F
from torchlinops.functional._unfold.nblocks import get_nblocks

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
        "small2d",
        pytest.param("small3d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_a2b(dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32
    ishape = (*spec["N"], *spec["shape"])

    x = torch.ones(prod(ishape)).to(dtype).reshape(ishape).requires_grad_(True)
    y = F.array_to_blocks(x, spec["block_size"], spec["stride"])
    y.abs().sum().backward()

    gradx = F.blocks_to_array(
        torch.ones_like(y),
        spec["shape"],
        spec["block_size"],
        spec["stride"],
    )
    assert torch.allclose(x.grad, gradx)


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
@pytest.mark.parametrize(
    "spec",
    [
        "small1d",
        "medium1d",
        "small2d",
        pytest.param("small3d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_b2a(dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32
    spec["nblocks"] = get_nblocks(spec["shape"], spec["block_size"], spec["stride"])
    ishape = (*spec["N"], *spec["nblocks"], *spec["block_size"])
    oshape = (*spec["N"], *spec["shape"])

    x = torch.ones(prod(ishape)).to(dtype).reshape(ishape).requires_grad_(True)
    y = F.blocks_to_array(x, spec["shape"], spec["block_size"], spec["stride"])
    y.abs().sum().backward()

    gradx = F.array_to_blocks(
        torch.ones_like(y),
        spec["block_size"],
        spec["stride"],
    )
    assert torch.allclose(x.grad, gradx)
