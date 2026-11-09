"""Slow, opt-in monitoring for compiled correctness, graph breaks, and speed.

Run with ``uv run pytest -m slow tests/test_compile_monitor.py -s`` after a
PyTorch upgrade. Correctness and graph-break counts are hard assertions; timing
is recorded and only guarded against a catastrophic regression.
"""

from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from benchmarks.compile.phases import _simple_chain
from benchmarks.compile.runner import capture_graph_breaks, median_time
from torchlinops import ArrayToBlocks, Chain, Dense, Interpolate, NUFFT, Sampling
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops.functional._unfold import unfold

pytestmark = pytest.mark.slow


@dataclass
class COMBO:
    name: str
    build: Callable[[], tuple[Callable, torch.Tensor]]
    device: str
    expected_breaks: int
    xfail: bool = False


def _dense(device: str):
    weight = torch.randn(256, 256, device=device)
    linop = Dense(weight, ("M", "N"), ("N",), ("M",))
    x = torch.randn(256, device=device)
    return lambda t: linop(t), x


def _chain(device: str):
    weight_a = torch.randn(256, 256, device=device)
    weight_b = torch.randn(256, 256, device=device)
    a = Dense(weight_a, ("M", "K"), ("K",), ("M",))
    b = Dense(weight_b, ("K", "N"), ("N",), ("K",))
    chain = Chain(b, a)
    x = torch.randn(256, device=device)
    return lambda t: chain(t), x


def _sampling(device: str):
    n, r, k, batch = 64, 32, 16, 4
    idx = torch.randint(0, n - 1, (r, k, 2), device=device)
    linop = Sampling.from_stacked_idx(idx, (n, n), ("R", "K"))
    x = torch.randn(batch, n, n, device=device)
    return lambda t: linop(t), x


def _nufft(device: str):
    locs = get_valid_locs((32,), (32, 32), 2, 4.0, "cpu", centered=True).to(device)
    linop = NUFFT(locs.clone(), (32, 32), output_shape=("R",), width=4.0, oversamp=1.25)
    linop._locs_orig = locs.clone()
    x = torch.randn(32, 32, dtype=torch.complex64, device=device)
    return lambda t: linop(t), x


def _interpolate(device: str):
    locs = get_valid_locs((32,), (32, 32), 2, 4.0, "cpu", centered=True).to(device)
    linop = Interpolate(locs, (32, 32), width=4.0, kernel="kaiser_bessel")
    x = torch.randn(32, 32, dtype=torch.complex64, device=device)
    return lambda t: linop(t), x


def _array_to_blocks(device: str):
    linop = ArrayToBlocks((32, 32), (16, 16), (8, 8))
    x = torch.randn(2, 32, 32, dtype=torch.complex64, device=device)
    return lambda t: linop(t), x


def _functional_unfold(device: str):
    x = torch.randn(2, 32, 32, dtype=torch.complex64, device=device)
    return lambda t: unfold(t, (16, 16), (8, 8)), x


def _pipeline(device: str):
    chain, x0 = _simple_chain(device)
    b = chain(x0).detach()
    chain_h = chain.H

    def step(x: torch.Tensor) -> torch.Tensor:
        for _ in range(20):
            x = x - 0.5 * chain_h(chain(x) - b)
        return x

    return step, x0


_COMBOS = [
    COMBO("Dense-fwd-cpu", lambda: _dense("cpu"), "cpu", expected_breaks=0),
    COMBO("Dense-fwd-cuda", lambda: _dense("cuda"), "cuda", expected_breaks=0),
    # Current develop compiles the wrapper without breaks for this chain.
    COMBO("Chain-fwd-cuda", lambda: _chain("cuda"), "cuda", expected_breaks=0),
    COMBO("Sampling-fwd-cuda", lambda: _sampling("cuda"), "cuda", expected_breaks=0),
    # Current develop compiles CUDA NUFFT without breaks.
    COMBO("NUFFT-fwd-cuda", lambda: _nufft("cuda"), "cuda", expected_breaks=0),
    COMBO("Interpolate-fwd-cpu", lambda: _interpolate("cpu"), "cpu", expected_breaks=0),
    COMBO(
        "ArrayToBlocks-fwd-cpu",
        lambda: _array_to_blocks("cpu"),
        "cpu",
        expected_breaks=0,
    ),
    COMBO(
        "functional.unfold-fwd-cpu",
        lambda: _functional_unfold("cpu"),
        "cpu",
        expected_breaks=0,
    ),
    # Current develop compiles the loop without wrapper graph breaks.
    COMBO(
        "pipeline-simple-default-cuda",
        lambda: _pipeline("cuda"),
        "cuda",
        expected_breaks=0,
    ),
]


def _param(combo: COMBO):
    marks = []
    if combo.device == "cuda":
        marks.append(pytest.mark.gpu)
        marks.append(
            pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="GPU is required but not available",
            )
        )
    if combo.xfail:
        marks.append(
            pytest.mark.xfail(
                strict=True, reason="known compile failure on this PyTorch"
            )
        )
    return pytest.param(combo, id=combo.name, marks=marks)


@pytest.mark.parametrize("combo", [_param(combo) for combo in _COMBOS])
def test_compile_monitor(combo: COMBO):
    target, input_ = combo.build()

    eager = target(input_)
    eager_time = median_time(
        lambda: target(input_),
        warmup=5,
        iters=20,
        device=combo.device,
    )

    torch._dynamo.reset()
    compiled = torch.compile(target)
    out = compiled(input_)
    torch.testing.assert_close(out, eager, rtol=1e-3, atol=1e-3)

    breaks, locations = capture_graph_breaks(target, input_, combo.name)
    assert breaks == combo.expected_breaks, (breaks, locations)

    compiled_time = median_time(
        lambda: compiled(input_),
        warmup=5,
        iters=20,
        device=combo.device,
    )
    speedup = eager_time / compiled_time if compiled_time else float("inf")
    print(
        f"[compile-monitor] combo={combo.name} device={combo.device} "
        f"breaks={breaks} speedup={speedup:.3f}"
    )
    assert compiled_time <= 2.0 * eager_time
