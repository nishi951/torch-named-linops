"""Compiled benchmark result collection and serialization."""

import json
import os
from pathlib import Path

from benchmarks._session import (
    collect_metadata,
    remove_owned_latest_files,
    write_patch_diff,
)
from benchmarks.compile.runner import PROTOCOLS, CellResult, effective_class


class CompiledBenchmarkSession:
    """Collects compiled benchmark results during a pytest session."""

    def __init__(self, config, level: str):
        self.config = config
        self.level = level
        config._compiled_results = []
        config._compiled_phases = {}
        config._compiled_level = level

    def record(self, result: CellResult) -> None:
        self.config._compiled_results.append(result_to_dict(result))

    def record_phase(self, kind: str, rows: list[dict]) -> None:
        self.config._compiled_phases[kind] = rows


def result_to_dict(result: CellResult) -> dict:
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


def write_compiled_results(config, latest_dir: Path) -> bool:
    level = getattr(config, "_compiled_level", None)
    results = getattr(config, "_compiled_results", None) or []
    phases = dict(getattr(config, "_compiled_phases", {}) or {})
    if not level or (not results and not phases):
        return False

    latest_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata()
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
        write_patch_diff(latest_dir)
    return True
