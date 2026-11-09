"""Shared helpers for benchmark result collection and archiving."""

import platform
import shutil
import subprocess  # nosec
from datetime import UTC, datetime
from pathlib import Path

import torch

NUM_WARMUP = 3

# Folders and files to check for a dirty working tree.
DIFF_PATHS = [
    "src/",
    "tests/",
    "benchmarks/",
    "uv.lock",
    "pyproject.toml",
]


def collect_metadata() -> dict:
    """Collect environment metadata for a benchmark run."""
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


def write_patch_diff(results_dir: Path) -> None:
    """Save the git diff when the working tree is dirty."""
    try:
        diff = subprocess.check_output(["git", "diff"]).decode()  # nosec
        if diff:
            (results_dir / "patch.diff").write_text(diff)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def archive_results(latest_dir: Path, metadata: dict) -> Path:
    """Copy the latest results to a date-SHA archive directory."""
    sha = metadata.get("commit_sha", "unknown")
    date = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    archive_name = f"{date}-{sha}"
    archive_dir = latest_dir.parent / archive_name
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    shutil.copytree(latest_dir, archive_dir)
    return archive_dir


def remove_owned_latest_files(latest_dir: Path, names: list[str]) -> None:
    """Remove only the named files from the latest results directory."""
    for name in names:
        path = latest_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
