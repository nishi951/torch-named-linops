"""Pytest configuration and session-scoped benchmark collector."""

import json
import os
import platform
import shutil
import subprocess  # nosec
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest
import torch

from torchlinops.utils.benchmark import CupyHandler, TorchHandler

NUM_WARMUP = 3

# Folders and files to check for diff.
DIFF_PATHS = [
    "src/",
    "tests/",
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
            try:
                if data_gen_fn is not None:
                    data = data_gen_fn()
                    fn(data)
                else:
                    fn()
            except Exception:
                pass

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
