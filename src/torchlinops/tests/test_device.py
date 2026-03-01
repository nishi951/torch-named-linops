from functools import partial
from itertools import product

import pytest
import torch

from torchlinops import Add, Chain, Concat, Dense, Dim, Stack, ToDevice
from torchlinops.utils import assert_gpus_overlap

# Linops that support parallel execution with multiple GPUs
PARALLELIZABLE_LINOPS = [
    Add,
    partial(Concat, odim="M"),
    partial(Stack, odim_and_idx=("M1", 0)),
]


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


def _slow_matmul_chain(N: int):
    weight = torch.randn(N, N)
    A = Dense(weight, weightshape=Dim("N1N"), ishape=Dim("N"), oshape=Dim("N1"))
    B = Dense(weight, weightshape=Dim("N2N1"), ishape=Dim("N1"), oshape=Dim("N2"))
    C = Dense(weight, weightshape=Dim("MN2"), ishape=Dim("N2"), oshape=Dim("M"))
    return C @ B @ A


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 GPUs are required for this test",
)
@pytest.mark.parametrize(
    "CombineOp,base_device",
    list(product(PARALLELIZABLE_LINOPS, [torch.device("cpu"), torch.device("cuda:0")])),
)
def test_multigpu_parallelism(CombineOp, base_device):
    gpu0 = torch.device("cuda:0")
    gpu1 = torch.device("cuda:1")

    N = 8192  # arbitrary

    # Input
    x = torch.randn(N, device=base_device)

    # Linop
    A1 = _slow_matmul_chain(N)
    A2 = _slow_matmul_chain(N)

    # True value (on cpu)
    OffDevice = CombineOp(A1, A2)
    y_true = OffDevice(x)

    # Move to GPU
    OnDevice = CombineOp(
        Chain(ToDevice(base_device, gpu0), A1.to(gpu0), ToDevice(gpu0, base_device)),
        Chain(ToDevice(base_device, gpu1), A2.to(gpu1), ToDevice(gpu1, base_device)),
    )

    # Warmup
    _ = OnDevice(x)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
    ) as prof:
        y = OnDevice(x)

        torch.cuda.synchronize(gpu0)
        torch.cuda.synchronize(gpu1)

    # Final device should be correct
    assert y.device.type == base_device.type

    # Parallelism
    assert_gpus_overlap(prof)

    # Correctness
    assert torch.allclose(y.cpu(), y_true)
