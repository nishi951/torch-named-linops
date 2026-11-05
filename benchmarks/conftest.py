"""Pytest configuration and session-scoped benchmark collector."""

import json
import os
import platform
import shutil
import subprocess  # nosec
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from benchmarks._compiled_cases import iter_specs, spec_id
from benchmarks._compiled_runner import PROTOCOLS, CellResult, effective_class
from torchlinops.utils.benchmark import CupyHandler, TorchHandler

NUM_WARMUP = 3

# Folders and files to check for diff.
DIFF_PATHS = [
    "src/",
    "tests/",
    "benchmarks/",
    "uv.lock",
    "pyproject.toml",
]


class BenchmarkSession:
    """Collects benchmark results during a pytest session."""

    def __init__(self, config, results_dir: Path):
        self.config = config
        self.results_dir = results_dir
        self.metadata = _collect_metadata()

    def run(
        self,
        name: str,
        fn,
        device: str = "cpu",
        min_run_time: float | None = None,
        label: str = "",
        sub_label: str = "",
        description: str = "",
        library: str = "torchlinops",
        data_gen_fn=None,
        size_name: str = "",
        problem_size: int | None = None,
        size_label: str = "",
    ):
        """Run a benchmark and store the result.

        Parameters
        ----------
        name : str
            Human-readable benchmark name.
        fn : callable
            Function to benchmark. If data_gen_fn is provided, fn should accept
            the generated data as an argument. Otherwise, fn takes no arguments.
        device : str
            "cpu" or "cuda".
        min_run_time : float, optional
            Minimum run time for benchmarking. If None, automatically determined.
        label, sub_label, description : str
            Grouping fields used by torch.utils.benchmark.Compare and the report.
        library : str
            "torchlinops" or "sigpy". Determines which benchmark backend to use.
        data_gen_fn : callable, optional
            If provided, this function regenerates data each iteration (not timed).
            The fn should accept the generated data as an argument.
        size_name : str
            Name of the problem size preset (e.g. "small", "medium", "large").
        problem_size : int, optional
            Numeric problem size (e.g. prod(grid_size)) used for scaling curves.
        size_label : str
            Human-readable size description (e.g. "64x64").
        """
        use_cupy = library == "sigpy" and device == "cuda"

        # Create appropriate handler
        if use_cupy:
            handler = CupyHandler()
        else:
            handler = TorchHandler(device=device, memory_snapshot_file=None)

        # Auto-determine min_run_time via pilot run
        if min_run_time is None:
            min_run_time = self._estimate_min_run_time(
                fn, device, use_cupy, data_gen_fn
            )

        min_runs_target = 10

        # Run main benchmark
        result = handler.blocked_autorange(
            fn,
            min_run_time=min_run_time,
            min_runs=min_runs_target,
            data_gen_fn=data_gen_fn,
            num_warmup=NUM_WARMUP,
        )

        # Build result dict
        result_dict = {
            "name": name,
            "library": library,
            "label": label or name,
            "sub_label": sub_label,
            "description": description,
            "device": device,
            "size_name": size_name,
            "problem_size": problem_size,
            "size_label": size_label,
            "mean_s": result.mean,
            "median_s": result.median,
            "iqr_s": result.iqr,
            "peak_mem_bytes": result.peak_mem_bytes,
            "num_runs": len(result.times),
            "number_per_run": result.number_per_run,
            "num_threads": torch.get_num_threads(),
        }
        self.config._benchmark_results.append(result_dict)

    def _estimate_min_run_time(
        self, fn, device: str, use_cupy: bool, data_gen_fn=None
    ) -> float:
        """Estimate min_run_time to ensure at least 10 runs.

        Does a quick pilot run to estimate per-call time, then sets
        min_run_time so that we get at least 10 runs within a reasonable
        total time (~5 seconds).
        """
        for _ in range(NUM_WARMUP):
            if data_gen_fn is not None:
                data = data_gen_fn()
                fn(data)
            else:
                fn()

        # Pilot run to estimate per-call time
        if use_cupy:
            import cupy

            cupy.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            if data_gen_fn is not None:
                data = data_gen_fn()
                fn(data)
            else:
                fn()
            cupy.cuda.Stream.null.synchronize()
            end = time.perf_counter()
        else:
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            if data_gen_fn is not None:
                data = data_gen_fn()
                fn(data)
            else:
                fn()
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

        per_call_time = end - start

        # Set min_run_time to ensure at least 10 runs
        if per_call_time > 0.01:  # > 10ms per call
            min_run_time = per_call_time * 1.5
        elif per_call_time > 0.001:  # > 1ms per call
            min_run_time = per_call_time * 5
        else:
            min_run_time = 0.1

        return max(0.01, min(0.5, min_run_time))


class CompiledBenchmarkSession:
    """Collects compiled benchmark results during a pytest session."""

    def __init__(self, config, level: str):
        self.config = config
        self.level = level
        config._compiled_results = []
        config._compiled_phases = {}
        config._compiled_level = level

    def record(self, result: CellResult) -> None:
        self.config._compiled_results.append(_result_to_dict(result))

    def record_phase(self, kind: str, rows: list[dict]) -> None:
        self.config._compiled_phases[kind] = rows


def _result_to_dict(result: CellResult) -> dict:
    spec = result.spec
    return {
        "op": spec.op,
        "category": spec.category,
        "method": spec.method,
        "device": spec.device,
        "dtype": str(spec.dtype),
        "level": spec.level,
        "config": spec.config,
        "status": result.status,
        "cell_class": result.cell_class,
        "effective_class": effective_class(
            result.status,
            result.cell_class,
            result.graph_breaks,
            result.speedup,
            result.degraded,
            spec.config,
        ),
        "graph_breaks": result.graph_breaks,
        "break_locations": list(result.break_locations),
        "recompiles": result.recompiles,
        "max_abs_diff": result.max_abs_diff,
        "max_rel_diff": result.max_rel_diff,
        "eager_time": result.eager_time,
        "compiled_time": result.compiled_time,
        "speedup": result.speedup,
        "degraded": result.degraded,
        "traceback_tail": result.traceback_tail,
        "notes": result.notes,
    }


def compiled_results_payload(results: list[dict], level: str, metadata: dict) -> dict:
    return {
        "metadata": {
            **metadata,
            "suite_level": level,
            "protocol": PROTOCOLS[level],
        },
        "level": level,
        "rows": results,
    }


def remove_owned_latest_files(latest_dir: Path, names: list[str]) -> None:
    """Remove only the named files from the latest results directory."""
    for name in names:
        path = latest_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _write_compiled_results(config, latest_dir: Path) -> bool:
    level = getattr(config, "_compiled_level", None)
    results = getattr(config, "_compiled_results", None) or []
    phases = dict(getattr(config, "_compiled_phases", {}) or {})
    if not level or (not results and not phases):
        return False

    latest_dir.mkdir(parents=True, exist_ok=True)
    metadata = _collect_metadata()
    payload = compiled_results_payload(results, level, metadata)
    payload["phases"] = phases

    filename = f"compiled-{level}.json"
    target = latest_dir / filename
    temp = latest_dir / f"{filename}.tmp"
    remove_owned_latest_files(latest_dir, [filename])
    try:
        temp.write_text(json.dumps(payload, indent=2))
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()

    if metadata.get("dirty") and not (latest_dir / "patch.diff").exists():
        _write_patch_diff(latest_dir)
    return True


def pytest_addoption(parser):
    parser.addoption(
        "--compiled-level",
        action="store",
        default="reduced",
        choices=("reduced", "full"),
        help="Compiled benchmark level to run (default: reduced).",
    )


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


def _collect_metadata():
    """Collect environment metadata for the benchmark run."""
    metadata = {
        "date": datetime.now(tz=UTC).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "os": platform.system(),
        "num_threads": torch.get_num_threads(),
    }

    try:
        metadata["commit_sha"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])  # nosec
            .decode()
            .strip()
        )
        metadata["full_sha"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])  # nosec
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        metadata["commit_sha"] = "unknown"
        metadata["full_sha"] = "unknown"

    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        metadata["gpu_name"] = None

    try:
        subprocess.check_output(["git", "diff", "--"] + DIFF_PATHS + ["--quiet"])  # nosec
        metadata["dirty"] = False
    except (subprocess.CalledProcessError, FileNotFoundError):
        metadata["dirty"] = True

    return metadata


def _write_patch_diff(results_dir: Path):
    """Save git diff if the working tree is dirty."""
    try:
        diff = subprocess.check_output(["git", "diff"]).decode()  # nosec
        if diff:
            (results_dir / "patch.diff").write_text(diff)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def _archive_results(latest_dir: Path, metadata: dict):
    """Copy the latest results to a date-SHA archive directory."""
    sha = metadata.get("commit_sha", "unknown")
    date = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    archive_name = f"{date}-{sha}"
    archive_dir = latest_dir.parent / archive_name
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    shutil.copytree(latest_dir, archive_dir)
    return archive_dir


@pytest.fixture(scope="session")
def benchmark_session(request):
    """Session-scoped fixture for collecting benchmark results."""
    benchmarks_dir = Path(__file__).parent
    results_dir = benchmarks_dir / "results" / "latest"
    results_dir.mkdir(parents=True, exist_ok=True)

    request.config._benchmark_results = []
    session = BenchmarkSession(request.config, results_dir)
    yield session


@pytest.fixture(scope="session")
def compiled_benchmark_session(request):
    """Session-scoped fixture for collecting compiled benchmark results."""
    level = request.config.getoption("compiled_level")
    return CompiledBenchmarkSession(request.config, level)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Write benchmark results to disk at the end of the session."""
    config = session.config
    results = getattr(config, "_benchmark_results", None)
    compiled_results = getattr(config, "_compiled_results", None)
    compiled_phases = getattr(config, "_compiled_phases", None)
    if not results and not compiled_results and not compiled_phases:
        return

    benchmarks_dir = Path(__file__).parent
    latest_dir = benchmarks_dir / "results" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    metadata = _collect_metadata()

    if results:
        # Remove only the files this eager suite owns.
        remove_owned_latest_files(
            latest_dir, ["results.json", "metadata.json", "patch.diff"]
        )
        (latest_dir / "results.json").write_text(json.dumps(results, indent=2))
        (latest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        if metadata.get("dirty"):
            _write_patch_diff(latest_dir)

    compiled_written = _write_compiled_results(config, latest_dir)

    if results or compiled_written:
        _archive_results(latest_dir, metadata)
