#!/usr/bin/env python
"""Generate benchmark documentation from the latest benchmark results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARKS_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARKS_DIR / "results" / "latest"
DOCS_DIR = Path(__file__).parent.parent / "docs" / "benchmarks"
ASSETS_DIR = DOCS_DIR / "assets"


def _format_time(seconds: float) -> str:
    """Format a time in seconds to a human-readable string."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _format_bytes(num_bytes) -> str:
    """Format bytes to a human-readable string."""
    if num_bytes is None:
        return "—"
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def _load_results():
    results_file = RESULTS_DIR / "results.json"
    metadata_file = RESULTS_DIR / "metadata.json"
    if not results_file.exists():
        return None, None
    results = json.loads(results_file.read_text())
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    return results, metadata


def _group_by_label(results):
    groups = {}
    for r in results:
        label = r.get("label") or r.get("name")
        groups.setdefault(label, []).append(r)
    return groups


def _build_metadata_section(metadata):
    lines = ["## Metadata", ""]
    lines.append(f"- **Date**: {metadata.get('date', 'unknown')}")
    lines.append(f"- **Commit**: `{metadata.get('commit_sha', 'unknown')}`")
    if metadata.get("dirty"):
        patch_path = "../../benchmarks/results/latest/patch.diff"
        lines.append(f"- **Working tree**: dirty ([patch.diff]({patch_path}))")
    else:
        lines.append("- **Working tree**: clean")
    lines.append(f"- **PyTorch**: {metadata.get('torch_version', 'unknown')}")
    lines.append(f"- **CUDA**: {metadata.get('cuda_version') or 'N/A'}")
    lines.append(f"- **GPU**: {metadata.get('gpu_name') or 'N/A'}")
    lines.append(f"- **Python**: {metadata.get('python_version', 'unknown')}")
    lines.append(f"- **OS**: {metadata.get('os', 'unknown')}")
    lines.append(f"- **Threads**: {metadata.get('num_threads', 'unknown')}")
    lines.append("")
    return "\n".join(lines)


def _build_table(results):
    header = "| Operation | Device | Mean | Median | IQR | Peak Memory |\n"
    separator = "|-----------|--------|------|--------|-----|-------------|\n"
    rows = []
    for r in results:
        iqr = _format_time(r["iqr_s"]) if r.get("iqr_s") is not None else "—"
        rows.append(
            f"| {r['name']} | {r['device']} | "
            f"{_format_time(r['mean_s'])} | {_format_time(r['median_s'])} | "
            f"{iqr} | {_format_bytes(r.get('peak_mem_bytes'))} |"
        )
    return header + separator + "\n".join(rows) + "\n"


def _build_markdown(results, metadata):
    lines = ["# Benchmarks", ""]
    lines.append("Performance benchmarks for torch-named-linops.")
    lines.append("")
    lines.append(_build_metadata_section(metadata))

    groups = _group_by_label(results)
    for label, group in sorted(groups.items()):
        lines.append(f"## {label}")
        lines.append("")
        lines.append(_build_table(group))
        lines.append("")

    lines.append("## Charts")
    lines.append("")
    lines.append("![Timing comparison](assets/timing_comparison.png)")
    lines.append("")
    lines.append("![Memory comparison](assets/memory_comparison.png)")
    lines.append("")

    return "\n".join(lines)


def _plot_timing(results):
    """Grouped bar chart of mean timing per benchmark, grouped by device."""
    names = sorted({r["name"] for r in results})
    devices = sorted({r["device"] for r in results})

    data = {name: {device: None for device in devices} for name in names}
    for r in results:
        data[r["name"]][r["device"]] = r["mean_s"]

    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 6))

    for i, device in enumerate(devices):
        values = [data[name][device] for name in names]
        ax.bar(
            [xi + i * width for xi in x],
            values,
            width,
            label=device,
        )

    ax.set_ylabel("Mean time (s)")
    ax.set_yscale("log")
    ax.set_title("Benchmark Timing Comparison")
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / "timing_comparison.png", dpi=150)
    plt.close(fig)


def _plot_memory(results):
    """Bar chart of peak GPU memory per GPU benchmark."""
    gpu_results = [
        r for r in results if r["device"] == "cuda" and r.get("peak_mem_bytes")
    ]
    if not gpu_results:
        # Create an empty placeholder chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No GPU memory data available", ha="center", va="center")
        ax.set_axis_off()
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(ASSETS_DIR / "memory_comparison.png", dpi=150)
        plt.close(fig)
        return

    names = [r["name"] for r in gpu_results]
    mem_bytes = [r["peak_mem_bytes"] for r in gpu_results]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 6))
    x = range(len(names))
    ax.bar(x, mem_bytes)
    ax.set_ylabel("Peak memory (bytes)")
    ax.set_yscale("log")
    ax.set_title("GPU Peak Memory Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / "memory_comparison.png", dpi=150)
    plt.close(fig)


def main():
    results, metadata = _load_results()
    if results is None:
        print(
            "No benchmark results found in benchmarks/results/latest/. Skipping report generation."
        )
        return

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    markdown = _build_markdown(results, metadata)
    (DOCS_DIR / "index.md").write_text(markdown)

    _plot_timing(results)
    _plot_memory(results)

    print(f"Benchmark report generated at {DOCS_DIR / 'index.md'}")


if __name__ == "__main__":
    main()
