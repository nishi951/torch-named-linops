"""Compiled benchmark entry point.

Rows are parametrized by ``benchmarks/conftest.py`` from the case registry,
according to ``--compiled-level``.
"""

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
