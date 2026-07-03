"""Pytest configuration and session-scoped benchmark collector."""

import json
import os
import platform
import shutil
import subprocess  # nosec
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest
import torch

from benchmarks._memory_tracker import benchmark_op


class BenchmarkSession:
    """Collects benchmark results during a pytest session."""

    def __init__(self, config, results_dir: Path):
        self.config = config
        self.results_dir = results_dir
        self.metadata = _collect_metadata()

    def _estimate_min_run_time(self, fn, device: str, use_cupy: bool) -> float:
        """Estimate min_run_time to ensure at least 10 runs.

        Does a quick pilot run to estimate per-call time, then sets
        min_run_time so that blocked_autorange will get at least 10 runs
        within a reasonable total time (~5 seconds).
        """
        import time

        # Warmup
        try:
            fn()
        except Exception:
            pass

        # Pilot run to estimate per-call time
        if use_cupy:
            import cupy

            cupy.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            fn()
            cupy.cuda.Stream.null.synchronize()
            end = time.perf_counter()
        else:
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

        per_call_time = end - start

        # Set min_run_time to ensure at least 10 runs
        # Strategy: min_run_time should be small enough that we get many runs
        # For slow functions (>10ms), use very small min_run_time
        # For fast functions, use larger min_run_time for stability
        if per_call_time > 0.01:  # > 10ms per call
            # Slow function: use small min_run_time to get many runs
            min_run_time = per_call_time * 1.5  # Each run has ~1-2 calls
        elif per_call_time > 0.001:  # > 1ms per call
            min_run_time = per_call_time * 5  # Each run has ~5 calls
        else:
            # Fast function: use larger min_run_time for stability
            min_run_time = 0.1

        # Clamp to reasonable range
        return max(0.01, min(0.5, min_run_time))

    def run(
        self,
        name: str,
        fn,
        device: str = "cpu",
        min_run_time: Optional[float] = None,
        label: str = "",
        sub_label: str = "",
        description: str = "",
        library: str = "torchlinops",
        data_gen_fn=None,
        size_name: str = "",
        problem_size: Optional[int] = None,
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

        # Auto-determine min_run_time to ensure at least 10 runs
        if min_run_time is None:
            if data_gen_fn is not None:
                # For separate data gen, we need to estimate based on fn alone
                min_run_time = self._estimate_min_run_time(
                    lambda: fn(data_gen_fn()), device, use_cupy
                )
            else:
                min_run_time = self._estimate_min_run_time(fn, device, use_cupy)

        # Run benchmark
        min_runs_target = 10

        if use_cupy:
            from benchmarks._cupy_benchmark import cupy_blocked_autorange

            effective_min_run_time = min_run_time

            data_gen_mean_s = None  # No longer tracking separately
            if data_gen_fn is not None:
                # Benchmark data gen separately just for reporting, not for subtraction
                gen_measurement = cupy_blocked_autorange(
                    data_gen_fn, min_run_time=0.1, min_runs=5
                )
                data_gen_mean_s = gen_measurement.mean

                # Wrap fn to use generated data
                def fn_with_data():
                    data = data_gen_fn()
                    return fn(data)

                measurement = cupy_blocked_autorange(
                    fn_with_data,
                    min_run_time=effective_min_run_time,
                    min_runs=min_runs_target,
                )
            else:
                measurement = cupy_blocked_autorange(
                    fn, min_run_time=effective_min_run_time, min_runs=min_runs_target
                )
            peak_mem = measurement.peak_mem_bytes
        else:
            data_gen_mean_s = None  # No longer tracking separately
            if data_gen_fn is not None:
                # Benchmark data gen separately just for reporting
                gen_measurement, _ = benchmark_op(
                    data_gen_fn, device=device, min_run_time=0.1, min_runs=5
                )
                data_gen_mean_s = gen_measurement.mean
                # Pass data_gen_fn to benchmark_op for separate timing
                measurement, peak_mem = benchmark_op(
                    fn,
                    device=device,
                    min_run_time=min_run_time,
                    min_runs=min_runs_target,
                    data_gen_fn=data_gen_fn,
                )
            else:
                measurement, peak_mem = benchmark_op(
                    fn,
                    device=device,
                    min_run_time=min_run_time,
                    min_runs=min_runs_target,
                )

        iqr = getattr(measurement, "iqr", None)
        result = {
            "name": name,
            "library": library,
            "label": label or name,
            "sub_label": sub_label,
            "description": description,
            "device": device,
            "size_name": size_name,
            "problem_size": problem_size,
            "size_label": size_label,
            "mean_s": measurement.mean,
            "data_gen_mean_s": data_gen_mean_s,
            "median_s": measurement.median,
            "iqr_s": iqr,
            "peak_mem_bytes": peak_mem,
            "num_runs": len(measurement.times),
            "number_per_run": measurement.number_per_run,
            "num_threads": torch.get_num_threads(),
        }
        self.config._benchmark_results.append(result)


def _collect_metadata():
    """Collect environment metadata for the benchmark run."""
    metadata = {
        "date": datetime.now().isoformat(),
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
        subprocess.check_output(["git", "diff", "--quiet"])  # nosec
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
    date = datetime.now().strftime("%Y-%m-%d")
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


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Write benchmark results to disk at the end of the session."""
    results = getattr(session.config, "_benchmark_results", None)
    if not results:
        return

    benchmarks_dir = Path(__file__).parent
    latest_dir = benchmarks_dir / "results" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    metadata = _collect_metadata()

    # Clear previous latest results (keep directory)
    for item in latest_dir.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    # Write results and metadata
    (latest_dir / "results.json").write_text(json.dumps(results, indent=2))
    (latest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if metadata.get("dirty"):
        _write_patch_diff(latest_dir)

    # Archive to date-SHA directory
    _archive_results(latest_dir, metadata)
