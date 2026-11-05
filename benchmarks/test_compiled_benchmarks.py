"""Compiled benchmark entry point.

Rows are parametrized by ``benchmarks/conftest.py`` from the case registry,
according to ``--compiled-level``.
"""

import pytest

from benchmarks._compiled_cases import CASE_BY_OP
from benchmarks._compiled_runner import PROTOCOLS, execute_cell


def test_compiled_row(compiled_spec, compiled_benchmark_session):
    case = CASE_BY_OP[compiled_spec.op]
    use_inner = compiled_spec.level == "inner" and case.inner_build is not None
    builder = case.inner_build if use_inner else case.build
    target, input_, note = builder(
        compiled_spec.device, compiled_spec.dtype, compiled_spec.method
    )
    protocol = PROTOCOLS[compiled_benchmark_session.level]
    result = execute_cell(compiled_spec, target, input_, **protocol)
    result.notes = " | ".join(filter(None, [note, result.notes]))
    compiled_benchmark_session.record(result)


def _require_full(request):
    if request.config.getoption("compiled_level") != "full":
        pytest.skip("full compiled level required")


@pytest.mark.benchmark
def test_recompile_phase(request, compiled_benchmark_session):
    _require_full(request)
    from benchmarks._compiled_phases import run_recompile_phase

    compiled_benchmark_session.record_phase("recompile", run_recompile_phase())


@pytest.mark.benchmark
def test_autograd_phase(request, compiled_benchmark_session):
    _require_full(request)
    from benchmarks._compiled_phases import run_autograd_phase

    compiled_benchmark_session.record_phase("autograd", run_autograd_phase())


@pytest.mark.benchmark
def test_pipeline_phase(request, compiled_benchmark_session):
    _require_full(request)
    from benchmarks._compiled_phases import run_pipeline_phase

    compiled_benchmark_session.record_phase("pipeline", run_pipeline_phase())
