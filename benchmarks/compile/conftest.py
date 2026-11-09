"""Pytest fixtures and parametrization for compiled benchmarks."""

import pytest
import torch

from benchmarks.compile.cases import iter_specs, spec_id
from benchmarks.compile.results import CompiledBenchmarkSession


@pytest.fixture(scope="session")
def compiled_benchmark_session(request):
    """Session-scoped fixture for collecting compiled benchmark results."""
    level = request.config.getoption("compiled_level")
    return CompiledBenchmarkSession(request.config, level)


def pytest_generate_tests(metafunc):
    if "compiled_spec" not in metafunc.fixturenames:
        return

    level = metafunc.config.getoption("compiled_level")
    params = []
    for spec in iter_specs(level):
        marks = [pytest.mark.benchmark]
        if spec.device == "cuda":
            marks.extend(
                [
                    pytest.mark.gpu,
                    pytest.mark.skipif(
                        not torch.cuda.is_available(),
                        reason="GPU is required but not available",
                    ),
                ]
            )
        params.append(pytest.param(spec, id=spec_id(spec), marks=marks))
    metafunc.parametrize("compiled_spec", params)
