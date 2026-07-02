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

    def run(
        self,
        name: str,
        fn,
        device: str = "cpu",
        min_run_time: Optional[float] = None,
        label: str = "",
        sub_label: str = "",
        description: str = "",
    ):
        """Run a benchmark and store the result.

        Parameters
        ----------
        name : str
            Human-readable benchmark name.
        fn : callable
            Function to benchmark (no arguments).
        device : str
            "cpu" or "cuda".
        min_run_time : float, optional
            Minimum run time for blocked_autorange. Defaults are chosen
            based on device.
        label, sub_label, description : str
            Grouping fields used by torch.utils.benchmark.Compare and the report.
        """
        measurement, peak_mem = benchmark_op(
            fn, device=device, min_run_time=min_run_time
        )

        iqr = getattr(measurement, "iqr", None)
        result = {
            "name": name,
            "label": label or name,
            "sub_label": sub_label,
            "description": description,
            "device": device,
            "mean_s": measurement.mean,
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
