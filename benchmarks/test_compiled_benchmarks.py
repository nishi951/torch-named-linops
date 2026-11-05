"""Compiled benchmark entry point.

Rows are parametrized by ``benchmarks/conftest.py`` from the case registry,
according to ``--compiled-level``.
"""

import traceback

import pytest
import torch

from benchmarks._compiled_cases import CASE_BY_OP
from benchmarks._compiled_runner import (
    PROTOCOLS,
    CellResult,
    execute_cell,
)

_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)


def run_compiled_row(spec, session):
    case = CASE_BY_OP[spec.op]
    use_inner = spec.level == "inner" and case.inner_build is not None
    builder = case.inner_build if use_inner else case.build

    try:
        target, input_, note = builder(spec.device, spec.dtype, spec.method)
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        result = CellResult(
            spec=spec,
            status="error",
            cell_class="fails_loud",
            graph_breaks=None,
            traceback_tail=traceback.format_exc().strip()[-800:],
            notes="case build failed",
        )
        session.record(result)
        return result

    protocol = PROTOCOLS[session.level]
    result = execute_cell(spec, target, input_, **protocol)
    result.notes = " | ".join(filter(None, [note, result.notes]))
    session.record(result)
    return result


def test_compiled_row(compiled_spec, compiled_benchmark_session):
    run_compiled_row(compiled_spec, compiled_benchmark_session)


def _require_full(request):
    if request.config.getoption("compiled_level") != "full":
        pytest.skip("full compiled level required")


@pytest.mark.benchmark
@pytest.mark.gpu
@_CUDA_REQUIRED
def test_recompile_phase(request, compiled_benchmark_session):
    _require_full(request)
    from benchmarks._compiled_phases import run_recompile_phase

    compiled_benchmark_session.record_phase("recompile", run_recompile_phase())


@pytest.mark.benchmark
@pytest.mark.gpu
@_CUDA_REQUIRED
def test_autograd_phase(request, compiled_benchmark_session):
    _require_full(request)
    from benchmarks._compiled_phases import run_autograd_phase

    compiled_benchmark_session.record_phase("autograd", run_autograd_phase())


@pytest.mark.benchmark
@pytest.mark.gpu
@_CUDA_REQUIRED
def test_pipeline_phase(request, compiled_benchmark_session):
    _require_full(request)
    from benchmarks._compiled_phases import run_pipeline_phase

    compiled_benchmark_session.record_phase("pipeline", run_pipeline_phase())
