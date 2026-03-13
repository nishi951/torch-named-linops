import logging
from copy import deepcopy
from functools import partial
from itertools import product

import pytest
import torch
from torch.cuda import default_stream
from torch.testing import assert_close

from torchlinops import (
    Add,
    Chain,
    Concat,
    Dense,
    DeviceSpec,
    Dim,
    NamedDimension as ND,
    Stack,
    ToDevice,
)
from torchlinops.linops.device import _gpu2gpu_transfer

logger = logging.getLogger("torchlinops")
# Linops that support parallel execution with multiple GPUs
PARALLELIZABLE_LINOPS = [
    partial(Add),
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


def _slow_matmul_chain(N: int, chain_length: int = 5):
    """Make arbitrary-length chains of random dense linops.

    Useful for making really slow linops.
    """
    in_dim = ND("N")
    out_dim = ND("M")
    weight = torch.randn(N, N)
    if chain_length == 1:
        return Dense(weight, Dim("MN"), Dim("N"), Dim("M"))
    next_out_dim = in_dim + 1
    A = Dense(
        weight,
        weightshape=(next_out_dim, in_dim),
        ishape=(in_dim,),
        oshape=(next_out_dim,),
    )
    in_dim = next_out_dim
    next_out_dim = next_out_dim + 1
    for i in range(chain_length - 1):
        weight = torch.randn(N, N)
        if i == chain_length - 2:
            next_out_dim = out_dim
        B = Dense(
            weight,
            weightshape=(next_out_dim, in_dim),
            ishape=(in_dim,),
            oshape=(next_out_dim,),
        )
        A = B @ A
        in_dim = next_out_dim
        next_out_dim = next_out_dim + 1
    return A


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 GPUs are required for this test",
)
def test_gpu2gpu_barebones():
    gpu0 = torch.device("cuda:0")
    gpu1 = torch.device("cuda:1")
    base_device = gpu0

    N = 8192  # arbitrary

    ### GPU1 -> GPU0
    # Input
    x = torch.ones(N, device=base_device)
    # Prepare
    # x = x.to(base_device)
    source_stream = default_stream(x.device)
    # transfer_stream = torch.cuda.Stream(x.device)
    transfer_stream = torch.cuda.Stream(gpu0)
    target_stream = default_stream(gpu1)

    # transfer_stream.wait_event(input_ready_event)
    transfer_stream.wait_stream(source_stream)
    with torch.cuda.stream(transfer_stream):
        y = x.to(gpu1, non_blocking=True)
        assert y[0].cpu() == x[0].cpu()
        x.record_stream(transfer_stream)
        y.record_stream(transfer_stream)
    target_stream.wait_stream(transfer_stream)

    torch.cuda.synchronize(gpu0)
    torch.cuda.synchronize(gpu1)

    # Correctness
    assert_close(y.cpu(), x.cpu())

    ### GPU1 -> GPU0

    # Event
    x1 = torch.ones(N, device=gpu1)
    source_stream = default_stream(x1.device)
    transfer_stream = torch.cuda.Stream(x1.device)
    target_stream = default_stream(gpu0)

    # transfer_stream.wait_event(input_ready_event)
    # Wait for stream, not events
    transfer_stream.wait_stream(source_stream)
    with torch.cuda.stream(transfer_stream):
        y1 = x1.to(gpu0, non_blocking=True)
        x1.record_stream(transfer_stream)
        y1.record_stream(transfer_stream)
    # y2.record_stream(target_stream)
    target_stream.wait_stream(transfer_stream)

    torch.cuda.synchronize(gpu0)
    torch.cuda.synchronize(gpu1)

    # Correctness
    assert_close(y1.cpu(), x1.cpu())


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 GPUs are required for this test",
)
def test_gpu2gpu_sanity_check():
    gpu0 = torch.device("cuda:0")
    gpu1 = torch.device("cuda:1")
    base_device = gpu0

    N = 8192  # arbitrary

    ### GPU1 -> GPU0
    # Input
    x = torch.randn(N)
    # Event
    device_spec = DeviceSpec(gpu0)
    device_spec.p2p_setup(gpu1)

    # Prepare
    x = x.to(base_device)
    source_stream = default_stream(x.device)
    transfer_stream = device_spec.transfer_stream
    target_stream = default_stream(gpu1)

    input_ready_event = source_stream.record_event()

    y = _gpu2gpu_transfer(
        x,
        target_stream,
        # gpu1,
        transfer_stream,
        # input_ready_event,
    )

    torch.cuda.synchronize(gpu0)
    torch.cuda.synchronize(gpu1)

    # Correctness
    assert_close(y.cpu(), x.cpu())

    ### GPU1 -> GPU0

    # Event
    device_spec = DeviceSpec(gpu1)
    device_spec.p2p_setup(gpu0)
    x = x.to(gpu1)
    source_stream = default_stream(x.device)
    transfer_stream = device_spec.transfer_stream
    target_stream = default_stream(gpu0)
    input_ready_event = source_stream.record_event()

    y2 = _gpu2gpu_transfer(
        x,
        # gpu0,
        target_stream,
        transfer_stream,
        # input_ready_event,
    )
    torch.cuda.synchronize(gpu0)
    torch.cuda.synchronize(gpu1)

    # Correctness
    assert_close(y2.cpu(), x.cpu())


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 GPUs are required for this test",
)
def test_gpu2gpu_oneway():
    gpu0 = torch.device("cuda:0")
    gpu1 = torch.device("cuda:1")
    base_device = gpu0

    N = 8192  # arbitrary

    # Input
    x = torch.randn(N)

    # Move to GPU
    x = x.to(base_device)
    # No threading involved
    # Include chain for simplicity
    S = Chain(ToDevice(base_device, gpu1))
    y = S(x)
    torch.cuda.synchronize(gpu0)
    torch.cuda.synchronize(gpu1)

    # Correctness
    assert_close(y.cpu(), x.cpu())

    x = x.to(gpu1)
    # No threading involved
    # Include chain for simplicity
    S2 = Chain(ToDevice(gpu1, base_device))
    y2 = S2(x)
    torch.cuda.synchronize(gpu0)
    torch.cuda.synchronize(gpu1)

    # Correctness
    assert_close(y2.cpu(), x.cpu())


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 GPUs are required for this test",
)
def test_gpu2gpu_roundtrip():
    gpu0 = torch.device("cuda:0")
    gpu1 = torch.device("cuda:1")
    base_device = gpu0

    N = 8192  # arbitrary

    # Input
    x = torch.ones(N)

    # Move to GPU
    x = x.to(base_device)
    torch.cuda.synchronize(base_device)
    # No threading involved
    S = Chain(
        ToDevice(base_device, gpu1),
        ToDevice(gpu1, base_device),
    )
    y = S(x)
    torch.cuda.synchronize(gpu0)
    torch.cuda.synchronize(gpu1)

    # Final device should be correct
    assert y.device.type == base_device.type

    # Correctness
    assert_close(y.cpu(), x.cpu())


def trace_handler(prof, OnDevice, base_device):
    prof.export_chrome_trace(
        f"./{type(OnDevice).__name__}_{base_device.type}_trace.json"
    )
    assert_gpus_overlap(prof, min_overlap_ms=0.0, min_overlap_ratio=0.1)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 GPUs are required for this test",
)
@pytest.mark.parametrize(
    "CombineOp,base_device",
    list(
        pytest.param(op, dev, id=f"{op.func.__name__}-{dev.type}")
        for op, dev in product(
            PARALLELIZABLE_LINOPS, [torch.device("cpu"), torch.device("cuda:0")]
        )
    ),
)
def test_multigpu_parallelism(CombineOp, base_device):
    gpu0 = torch.device("cuda:0")
    gpu1 = torch.device("cuda:1")

    N = 8192  # arbitrary

    # Input
    x = torch.randn(N)

    # Linop
    A1 = _slow_matmul_chain(N, 3)
    A2 = _slow_matmul_chain(N, 3)

    # True value (on cpu)
    # OffDevice = CombineOp(A1, A2)
    # y_true = OffDevice(x)
    logger.info("Building OffDevice linop")
    OffDevice = CombineOp(A1, A2)
    y_true = OffDevice(x)

    # Move to GPU
    x = x.to(base_device)
    logger.info("Building OnDevice linop")
    OnDevice = CombineOp(
        Chain(
            ToDevice(base_device, gpu0, ioshape=A1.ishape),
            deepcopy(A1).to(gpu0),
            ToDevice(gpu0, base_device, ioshape=A1.oshape),
        ),
        Chain(
            ToDevice(base_device, gpu1, ioshape=A2.ishape),
            deepcopy(A2).to(gpu1),
            ToDevice(gpu1, base_device, ioshape=A2.oshape),
        ),
    )

    # Warmup
    _ = OnDevice(x)

    wait, warmup, active = 2, 2, 1
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        schedule=torch.profiler.schedule(
            wait=wait,  # skip first 2 steps entirely
            warmup=warmup,  # record but discard next 2
            active=active,  # actually capture these
        ),
        on_trace_ready=partial(
            trace_handler, OnDevice=OnDevice, base_device=base_device
        ),
    ) as prof:
        for _ in range(wait + warmup + active):
            y = OnDevice(x)

            torch.cuda.synchronize(gpu0)
            torch.cuda.synchronize(gpu1)
            torch.cuda.synchronize()  # Synchronize all
            prof.step()

    # Final device should be correct
    assert y.device.type == base_device.type

    # Correctness
    assert_close(y.cpu(), y_true, atol=1e1, rtol=1e0)


def _get_cuda_kernel_events(prof):
    """
    Extract CUDA kernel events from a torch.profiler profile object.
    """
    return [
        evt
        for evt in prof.events()
        if "CUDA" in str(evt.device_type) and evt.device_time_total > 0
    ]


def _merge_intervals(intervals):
    """
    Merge overlapping intervals.
    intervals: list of (start_ns, end_ns)
    Returns merged list.
    """
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _compute_overlap(intervals_a, intervals_b):
    """
    Compute total overlap (ns) between two interval lists.
    Assumes intervals are already merged.
    """
    i = j = 0
    total = 0

    while i < len(intervals_a) and j < len(intervals_b):
        a_start, a_end = intervals_a[i]
        b_start, b_end = intervals_b[j]

        start = max(a_start, b_start)
        end = min(a_end, b_end)

        if start < end:
            total += end - start

        # Advance the interval that ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1

    return total


def assert_gpus_overlap(
    prof,
    *,
    min_overlap_ms: float = 5.0,
    min_overlap_ratio: float | None = None,
):
    """
    Assert that at least two GPUs executed concurrently.

    Args:
        prof: torch.profiler.profile object
        min_overlap_ms: minimum absolute overlap required (default 5ms)
        min_overlap_ratio: optional minimum fraction of smaller GPU runtime
                           that must overlap (e.g. 0.2 for 20%)
    """

    kernels = _get_cuda_kernel_events(prof)

    if not kernels:
        raise AssertionError("No CUDA kernel events found in profiler trace")

    # Group intervals by device
    by_device = {}
    for evt in kernels:
        start = evt.time_range.start
        end = evt.time_range.end
        by_device.setdefault(evt.device_index, []).append((start, end))

    if len(by_device) < 2:
        raise AssertionError("Need kernels from at least two GPUs")

    # Merge intervals per device
    merged = {dev: _merge_intervals(intervals) for dev, intervals in by_device.items()}

    devices = sorted(merged.keys())

    # Check all device pairs
    found_valid_overlap = False
    diagnostic = []

    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            d0, d1 = devices[i], devices[j]

            overlap_ns = _compute_overlap(
                merged[d0],
                merged[d1],
            )

            total0 = sum(e - s for s, e in merged[d0])
            total1 = sum(e - s for s, e in merged[d1])
            smaller_total = min(total0, total1)

            overlap_ms = overlap_ns / 1e6
            ratio = overlap_ns / smaller_total if smaller_total > 0 else 0.0

            diagnostic.append(
                f"GPU {d0} vs {d1}: "
                f"{overlap_ms:.3f} ms overlap "
                f"({ratio:.2%} of smaller runtime)"
            )

            abs_ok = overlap_ms >= min_overlap_ms
            ratio_ok = True if min_overlap_ratio is None else ratio >= min_overlap_ratio

            if abs_ok and ratio_ok:
                found_valid_overlap = True

    if not found_valid_overlap:
        diag_str = "\n".join(diagnostic)
        raise AssertionError(
            "No sufficient GPU concurrency detected.\n"
            f"Criteria: min_overlap_ms={min_overlap_ms}, "
            f"min_overlap_ratio={min_overlap_ratio}\n"
            f"Diagnostics:\n{diag_str}"
        )
