from copy import deepcopy

import pytest
import torch

from torchlinops import BatchSpec, Dense, create_batched_linop, split_linop


def test_split_linop():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    device = "cpu"
    A = Dense(weight, weightshape, ishape, oshape)
    linops, in_slc, out_slc = split_linop(A, {"N": 2, "M": 1})

    # tile indices
    n, m = 1, 2
    # Input
    x_n = torch.randn(B, 2)
    y_m = linops[n, m](x_n)
    # True operator
    A_mn = Dense(weight[:, m : m + 1, 2 * n : 2 * (n + 1)], weightshape, ishape, oshape)
    y_m_ref = A_mn(x_n)
    assert torch.allclose(y_m, y_m_ref)


def test_create_batched_linop():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    device = "cpu"
    A = Dense(weight, weightshape, ishape, oshape)

    Abatch = create_batched_linop(A, BatchSpec(dict(N=2, M=1)))
    x = torch.randn(B, N)
    assert Abatch(x).allclose(A(x), rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_create_batched_linop_multi_device():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    device = "cpu"
    A = Dense(weight, weightshape, ishape, oshape)

    Abatch = create_batched_linop(
        A,
        [
            BatchSpec(
                dict(N=2),
                device_matrix=[torch.device("cpu"), torch.device("cuda:0")],
                base_device=torch.device("cpu"),
            ),
            BatchSpec(dict(M=1)),
        ],
    )
    for _ in range(10):
        # Fuzzing with multiple retries
        x = torch.randn(B, N)
        Abatch_x = Abatch(x)
        Ax = A(x)
        assert Abatch_x.allclose(Ax, rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="At least 2 GPUs are required but not available",
)
def test_create_batched_linop_multi_device_gpu_only():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    A = Dense(weight, weightshape, ishape, oshape).to(torch.device("cuda:0"))

    Abatch = create_batched_linop(
        A,
        [
            BatchSpec(
                dict(N=2),
                device_matrix=[torch.device("cuda:0"), torch.device("cuda:1")],
                base_device=torch.device("cuda:0"),
            ),
            BatchSpec(dict(M=1)),
        ],
    )
    for _ in range(10):
        # Fuzzing with multiple retries
        x = torch.randn(B, N, device=torch.device("cuda:0"))
        Abatch_x = Abatch(x)
        Ax = A(x)
        assert Abatch_x.allclose(Ax, rtol=1e-3)


def test_batchspec_not_mutated_by_create_batched_linop():
    """Verify that create_batched_linop does not mutate shared BatchSpec objects.

    Regression test: BatchSpec objects in batch_specs[1:] were mutated in-place
    during the first tile's recursion, causing subsequent tiles to inherit the
    first tile's device settings via default_to(...) returning the stale value.
    """
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    A = Dense(weight, weightshape, ishape, oshape)

    inner_spec = BatchSpec(dict(M=1))
    assert inner_spec.base_device is None
    assert inner_spec.device_matrix is None

    _Abatch = create_batched_linop(
        A,
        [
            BatchSpec(dict(N=2)),
            inner_spec,
        ],
    )

    # After create_batched_linop, inner_spec should NOT have been mutated
    assert inner_spec.base_device is None, (
        f"BatchSpec.base_device was mutated to {inner_spec.base_device}"
    )
    assert inner_spec.device_matrix is None, (
        f"BatchSpec.device_matrix was mutated to {inner_spec.device_matrix}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_batched_linop_weight_integrity_multi_device():
    """Verify sub-tile weights have correct data after multi-device batching.

    Regression test: Due to BatchSpec mutation, inner recursion for tiles
    targeting a non-default device would use the wrong device, potentially
    leaving weights as uninitialized zeros on the target device.
    """
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    A = Dense(weight, weightshape, ishape, oshape)

    batch_specs = [
        BatchSpec(
            dict(N=2),
            device_matrix=[torch.device("cpu"), torch.device("cuda:0")],
            base_device=torch.device("cpu"),
        ),
        BatchSpec(dict(M=1)),
    ]
    batch_specs_before = deepcopy(batch_specs)
    Abatch = create_batched_linop(A, batch_specs)

    # Verify the inner BatchSpec was not mutated
    assert batch_specs[1].base_device == batch_specs_before[1].base_device, (
        f"Inner BatchSpec.base_device mutated: {batch_specs_before[1].base_device} -> {batch_specs[1].base_device}"
    )

    # Walk the tree and check every Dense leaf's weight is non-zero
    def check_weights(linop, path="root"):
        if isinstance(linop, Dense):
            assert linop.weight.abs().sum() > 0, (
                f"Dense weight is all zeros at {path} on device {linop.weight.device}"
            )
            return
        # Recurse into children (Concat, Chain, Add, etc.)
        if hasattr(linop, "linops"):
            for i, child in enumerate(linop.linops):
                check_weights(child, path=f"{path}[{i}]")
        for name, child in linop.named_children():
            if not hasattr(linop, "linops"):  # Avoid double-visiting
                check_weights(child, path=f"{path}.{name}")

    check_weights(Abatch)

    # Verify end-to-end correctness
    for _ in range(10):
        x = torch.randn(B, N)
        assert Abatch(x).allclose(A(x), rtol=1e-3), (
            "Batched linop output does not match original"
        )
