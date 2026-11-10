"""Pytest fixtures and parametrization for compiled benchmarks."""

import pytest
import torch

from benchmarks.compile.cases import iter_specs, spec_id
from benchmarks.compile.results import CompiledBenchmarkSession


@pytest.fixture(scope="session")
def compiled_benchmark_session(request):
    """Provide the session-scoped compiled result collector.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest request used to read ``--compiled-level``.

    Returns
    -------
    CompiledBenchmarkSession
        Collector for the selected suite level.
    """
    level = request.config.getoption("compiled_level")
    return CompiledBenchmarkSession(request.config, level)


def pytest_generate_tests(metafunc):
    """Parametrize compiled benchmark cells for the selected level.

    Reads ``--compiled-level``, expands the case registry with ``iter_specs``,
    and parametrizes the ``compiled_spec`` fixture. Cells are marked
    ``benchmark``; CUDA cells also get ``gpu`` and a CUDA-availability skip.

    Parameters
    ----------
    metafunc : pytest.Metafunc
        Pytest metafunction used to register parametrization.
    """
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
