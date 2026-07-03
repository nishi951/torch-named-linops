#!/usr/bin/env python
"""Generate benchmark documentation from the latest benchmark results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BENCHMARKS_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARKS_DIR / "results" / "latest"
DOCS_DIR = Path(__file__).parent.parent / "docs" / "benchmarks"
ASSETS_DIR = DOCS_DIR / "assets"

LIBRARY_COLORS = {
    "torchlinops": "#4C72B0",
    "torchlinops (linop)": "#7FB3D5",
    "sigpy": "#DD8452",
}

LIBRARY_MARKERS = {
    "torchlinops": "o",
    "torchlinops (linop)": "s",
    "sigpy": "^",
}

DIRECTION_LINESTYLES = {
    "forward": "-",
    "adjoint": "--",
}


def _format_time(seconds):
    if seconds is None:
        return "—"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _format_bytes(num_bytes):
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
    header = "| Operation | Size | Library | Device | Adj. Mean | Data Gen | Mean (total) | Median | IQR | Peak Memory |\n"
    separator = "|-----------|------|---------|--------|-----------|----------|--------------|--------|-----|-------------|\n"
    rows = []
    for r in results:
        iqr = _format_time(r.get("iqr_s")) if r.get("iqr_s") is not None else "—"
        adj = (
            _format_time(r.get("adjusted_mean_s"))
            if r.get("adjusted_mean_s") is not None
            else "—"
        )
        gen = (
            _format_time(r.get("data_gen_mean_s"))
            if r.get("data_gen_mean_s") is not None
            else "—"
        )
        rows.append(
            f"| {r['name']} | {r.get('size_label', '')} | {r.get('library', 'torchlinops')} | {r['device']} | "
            f"{adj} | {gen} | {_format_time(r['mean_s'])} | "
            f"{_format_time(r['median_s'])} | {iqr} | {_format_bytes(r.get('peak_mem_bytes'))} |"
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

    lines.append("## Bar Charts")
    lines.append("")
    for size_name in ["small", "medium", "large"]:
        lines.append(f"### {size_name.capitalize()}")
        lines.append("")
        lines.append(
            f"![Timing bar chart ({size_name})](assets/timing_{size_name}.png)"
        )
        lines.append("")

    lines.append("## Scaling Curves")
    lines.append("")
    lines.append("![Timing scaling](assets/scaling_time.png)")
    lines.append("")
    lines.append("![Memory scaling](assets/scaling_memory.png)")
    lines.append("")

    return "\n".join(lines)


def _get_time(r):
    """Get adjusted mean time, falling back to raw mean."""
    t = r.get("adjusted_mean_s")
    if t is None:
        t = r.get("mean_s")
    return t


def _plot_bar_chart(results, size_name):
    """Grouped bar chart for a single problem size."""
    size_results = [r for r in results if r.get("size_name") == size_name]
    if not size_results:
        return

    # Build composite labels: "Name\n(device)"
    entries = {}
    for r in size_results:
        key = f"{r['name']}\n({r['device']})"
        library = r.get("library", "torchlinops")
        entries.setdefault(key, {})[library] = _get_time(r)

    names = sorted(entries.keys())
    libraries = sorted({lib for d in entries.values() for lib in d})

    n_libraries = len(libraries)
    width = 0.8 / max(n_libraries, 1)
    x = np.arange(len(names))

    fig, ax = plt.subplots(
        figsize=(max(10, len(names) * 1.0), max(6, len(names) * 0.4))
    )

    for i, library in enumerate(libraries):
        values = [entries[name].get(library, 0) for name in names]
        color = LIBRARY_COLORS.get(library, None)
        ax.bar(x + i * width, values, width, label=library, color=color)

    ax.set_ylabel("Adjusted mean time (s)")
    ax.set_yscale("log")
    ax.set_title(f"Benchmark Timing Comparison — {size_name.capitalize()}")
    ax.set_xticks(x + width * (n_libraries - 1) / 2)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / f"timing_{size_name}.png", dpi=150)
    plt.close(fig)


def _plot_scaling_time(results):
    """Scaling curves: adjusted mean time vs problem size, per operation type.

    Grid of subplots: 2 columns (CPU, GPU) × N rows (operation types).
    Each subplot has lines per library × direction.
    """
    # Group by (label, device) to build per-operation subplots
    labels = sorted({r.get("label", "") for r in results})
    devices = ["cpu", "cuda"]

    n_rows = len(labels)
    n_cols = len(devices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    for row_idx, label in enumerate(labels):
        for col_idx, device in enumerate(devices):
            ax = axes[row_idx][col_idx]
            device_results = [
                r
                for r in results
                if r.get("label") == label and r.get("device") == device
            ]
            if not device_results:
                ax.set_visible(False)
                continue

            # Group by (library, description) → list of (problem_size, time)
            series = {}
            for r in device_results:
                key = (r.get("library", "torchlinops"), r.get("description", ""))
                ps = r.get("problem_size")
                t = _get_time(r)
                if ps is not None and t is not None:
                    series.setdefault(key, []).append((ps, t))

            for key, points in sorted(series.items()):
                library, direction = key
                points.sort()
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                color = LIBRARY_COLORS.get(library, None)
                marker = LIBRARY_MARKERS.get(library, "o")
                ls = DIRECTION_LINESTYLES.get(direction, "-")
                label_str = f"{library} ({direction})"
                ax.plot(
                    xs,
                    ys,
                    ls,
                    marker=marker,
                    color=color,
                    label=label_str,
                    markersize=5,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Problem size (elements)")
            ax.set_ylabel("Adjusted mean time (s)")
            ax.set_title(f"{label} ({device})")
            ax.legend(fontsize=7)
            ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.suptitle("Timing Scaling Curves", fontsize=14, y=1.0)
    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / "scaling_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scaling_memory(results):
    """Scaling curves: peak GPU memory vs problem size, per operation type.

    Single column, one row per operation type. GPU only.
    """
    gpu_results = [
        r for r in results if r.get("device") == "cuda" and r.get("peak_mem_bytes")
    ]
    if not gpu_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No GPU memory data available", ha="center", va="center")
        ax.set_axis_off()
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(ASSETS_DIR / "scaling_memory.png", dpi=150)
        plt.close(fig)
        return

    labels = sorted({r.get("label", "") for r in gpu_results})

    n_rows = len(labels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), squeeze=False)

    for row_idx, label in enumerate(labels):
        ax = axes[row_idx][0]
        label_results = [r for r in gpu_results if r.get("label") == label]

        series = {}
        for r in label_results:
            key = (r.get("library", "torchlinops"), r.get("description", ""))
            ps = r.get("problem_size")
            mem = r.get("peak_mem_bytes")
            if ps is not None and mem is not None:
                series.setdefault(key, []).append((ps, mem))

        for key, points in sorted(series.items()):
            library, direction = key
            points.sort()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            color = LIBRARY_COLORS.get(library, None)
            marker = LIBRARY_MARKERS.get(library, "o")
            ls = DIRECTION_LINESTYLES.get(direction, "-")
            label_str = f"{library} ({direction})"
            ax.plot(
                xs, ys, ls, marker=marker, color=color, label=label_str, markersize=5
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Problem size (elements)")
        ax.set_ylabel("Peak memory (bytes)")
        ax.set_title(f"{label} (GPU)")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.suptitle("GPU Memory Scaling Curves", fontsize=14, y=1.0)
    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / "scaling_memory.png", dpi=150, bbox_inches="tight")
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

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for size_name in ["small", "medium", "large"]:
        _plot_bar_chart(results, size_name)

    _plot_scaling_time(results)
    _plot_scaling_memory(results)

    print(f"Benchmark report generated at {DOCS_DIR / 'index.md'}")


if __name__ == "__main__":
    main()
