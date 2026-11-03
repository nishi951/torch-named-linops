"""Render the machine-generated part of the investigation report.

Writes `report-auto.md`; the curated `report.md` is hand-finished from it
so a re-run can never overwrite the narrative.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

REPORT_COLUMNS = [
    "phase",
    "op",
    "method",
    "device",
    "dtype",
    "level",
    "config",
    "class",
    "breaks",
    "rel_diff",
    "speedup",
]


def effective_class(cell: dict) -> str:
    stored = cell.get("effective_class")
    if stored:
        return stored
    if cell.get("status") == "n_a":
        return "n_a"
    if cell.get("cell_class") != "works":
        return cell["cell_class"]
    if (cell.get("graph_breaks") or 0) > 0:
        return "works_with_breaks"
    if cell.get("speedup") is not None and cell["speedup"] < 0.9:
        return "compiles_slower"
    return "works"


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _error_signature(tail: str | None) -> str:
    if not tail:
        return ""
    for line in reversed(tail.splitlines()):
        stripped = line.strip()
        if any(word in stripped for word in ("Error:", "Unsupported:", "Exception:")):
            return stripped[:240]
    return tail.strip().splitlines()[-1][:240] if tail.strip() else ""


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def render_report(results_path: Path, out_path: Path) -> None:
    data = json.loads(results_path.read_text())
    cells = data.get("cells", [])
    recompiles = data.get("recompile", [])
    autograd = data.get("autograd", [])
    pipeline = data.get("pipeline", [])
    deep = [cell for cell in cells if cell.get("phase") == "deep"]
    smoke = [cell for cell in cells if cell.get("phase") != "deep"]

    lines: list[str] = []
    lines += [
        "# torch.compile compatibility investigation — machine report",
        "",
        "Generated from `results/results.json` by `report.py`.",
        "Environment: PyTorch 2.15.0+cu132 (bundled triton), NVIDIA RTX 3090, CUDA 13.2.",
        "",
        "## Overview",
        "",
        f"- Sweep cells: {len(cells)} ({len(smoke)} smoke, {len(deep)} deep-dive)",
        f"- Recompile checks: {len(recompiles)}",
        f"- Autograd checks: {len(autograd)}",
        f"- Pipeline rows: {len(pipeline)}",
        "",
    ]

    class_counts = Counter(effective_class(cell) for cell in cells)
    lines += ["### Effective class counts", ""]
    lines += _table(
        ["class", "cells"],
        [[name, class_counts[name]] for name in sorted(class_counts)],
    )
    lines.append("")

    silent = [cell for cell in cells if cell.get("cell_class") == "fails_silent"]
    lines += ["### Silent numeric mismatches", ""]
    if silent:
        lines += _table(
            ["op", "level", "config", "device", "max_rel_diff"],
            [
                [
                    cell["op"],
                    cell["level"],
                    cell["config"],
                    cell["device"],
                    _fmt(cell.get("max_rel_diff")),
                ]
                for cell in silent
            ],
        )
        lines += ["", "**RED FLAG: compiled output disagreed with eager.**"]
    else:
        lines.append(
            "None: every compiled cell that ran agreed with eager within 1e-3 rel."
        )
    lines.append("")

    lines += ["## Wrapper vs inner level", ""]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["op"], cell["level"])].append(cell)
    summary_rows = []
    for (op, level), group in sorted(grouped.items()):
        classes = Counter(effective_class(cell) for cell in group)
        speeds = [cell["speedup"] for cell in group if cell.get("speedup") is not None]
        summary_rows.append(
            [
                op,
                level,
                len(group),
                classes.get("works", 0),
                classes.get("works_with_breaks", 0),
                classes.get("compiles_slower", 0),
                classes.get("fails_loud", 0),
                _fmt(median(speeds)) if speeds else "-",
                max((cell.get("graph_breaks") or 0) for cell in group),
            ]
        )
    lines += _table(
        [
            "op",
            "level",
            "cells",
            "works",
            "with_breaks",
            "slower",
            "fails",
            "median_speedup",
            "max_breaks",
        ],
        summary_rows,
    )
    lines.append("")

    lines += ["## Full cell matrix", ""]
    matrix_rows = []
    for cell in sorted(
        cells,
        key=lambda c: (
            c.get("phase", ""),
            c["op"],
            c["level"],
            c["method"],
            c["device"],
            c["config"],
        ),
    ):
        matrix_rows.append(
            [
                cell.get("phase", "smoke"),
                cell["op"],
                cell["method"],
                cell["device"],
                cell["dtype"],
                cell["level"],
                cell["config"],
                effective_class(cell),
                cell.get("graph_breaks", 0),
                _fmt(cell.get("max_rel_diff")),
                _fmt(cell.get("speedup")),
            ]
        )
    lines += _table(REPORT_COLUMNS, matrix_rows)
    lines.append("")

    lines += ["## Speedup table (default config, sorted)", ""]
    speed_rows = []
    for cell in sorted(
        [c for c in cells if c.get("speedup") is not None],
        key=lambda c: c["speedup"],
        reverse=True,
    ):
        speed_rows.append(
            [
                cell["op"],
                cell["level"],
                cell["method"],
                cell["device"],
                cell["dtype"],
                _fmt(cell["speedup"]),
                effective_class(cell),
            ]
        )
    lines += _table(
        ["op", "level", "method", "device", "dtype", "speedup", "class"], speed_rows
    )
    lines.append("")

    lines += ["## Recompilation checks", ""]
    if recompiles:
        lines += _table(
            [
                "op",
                "level",
                "device",
                "recompiles",
                "agrees_after",
                "max_rel_diff",
                "shapes",
                "notes",
            ],
            [
                [
                    record["op"],
                    record["level"],
                    record["device"],
                    _fmt(record.get("recompiles")),
                    _fmt(record.get("agrees_after")),
                    _fmt(record.get("max_rel_diff")),
                    record.get("shapes"),
                    (record.get("notes") or "")[:120],
                ]
                for record in recompiles
            ],
        )
    else:
        lines.append("Not run.")
    lines.append("")

    lines += ["## Autograd checks (single-op backward)", ""]
    if autograd:
        lines += _table(
            ["op", "level", "class", "max_rel_diff", "notes"],
            [
                [
                    row["op"],
                    row["level"],
                    row["cell_class"],
                    _fmt(row.get("max_rel_diff")),
                    (row.get("notes") or "")[:120],
                ]
                for row in autograd
            ],
        )
    else:
        lines.append("Not run.")
    lines.append("")

    lines += ["## Pipeline checks", ""]
    if pipeline:
        lines += _table(
            [
                "pipeline",
                "level",
                "config",
                "class",
                "speedup",
                "breaks",
                "notes",
            ],
            [
                [
                    row["op"],
                    row["level"],
                    row["config"],
                    row["cell_class"],
                    _fmt(row.get("speedup")),
                    row.get("graph_breaks", 0),
                    (row.get("notes") or "")[:120],
                ]
                for row in pipeline
            ],
        )
    else:
        lines.append("Not run.")
    lines.append("")

    lines += ["## Deep-dive observations (auto-collected)", ""]
    signatures: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for cell in deep:
        if cell.get("cell_class") == "fails_loud":
            signatures[(cell["op"], cell["level"], cell["config"])][
                _error_signature(cell.get("traceback_tail"))
            ] += 1
    for (op, level, config), counter in sorted(signatures.items()):
        lines.append(f"- **{op}** ({level}, {config}):")
        for signature, count in counter.most_common(3):
            lines.append(f"  - x{count}: `{signature}`")
    lines.append("")
    lines.append("Mechanism notes (hand-written): TBD")
    lines.append("")

    lines += ["## Implicated root causes (hand-written)", "", "TBD", ""]

    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    base = Path(__file__).parent
    render_report(base / "results" / "results.json", base / "report-auto.md")
    print(f"wrote {base / 'report-auto.md'}")
