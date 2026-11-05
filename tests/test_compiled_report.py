import json
from pathlib import Path

from benchmarks._compiled_report import build_compiled_markdown, load_compiled_results


def _payload(level="reduced", config="default"):
    return {
        "level": level,
        "metadata": {
            "suite_level": level,
            "torch_version": "2.15.0",
            "date": "2026-11-05",
        },
        "rows": [
            {
                "op": "Dense",
                "method": "fwd",
                "device": "cuda",
                "dtype": "torch.float32",
                "level": "wrapper",
                "config": config,
                "status": "ok",
                "cell_class": "works",
                "effective_class": "works",
                "graph_breaks": 0,
                "max_rel_diff": 0.0,
                "eager_time": 2.0,
                "compiled_time": 1.0,
                "speedup": 2.0,
                "traceback_tail": None,
                "notes": "",
            }
        ],
        "phases": {},
    }


def test_compiled_markdown_includes_graph_break_column():
    md = build_compiled_markdown(_payload(), "reduced")
    assert "Graph breaks" in md
    assert "Dense" in md
    assert "reduced" in md


def test_compiled_markdown_without_data():
    md = build_compiled_markdown(None, None)
    assert "No compiled benchmark data" in md


def test_reduced_page_marks_phases_as_full_only():
    md = build_compiled_markdown(_payload(), "reduced")
    assert "full suite" in md.lower()


def test_full_page_renders_phase_sections():
    data = _payload("full")
    data["phases"] = {
        "recompile": [{"op": "Dense", "level": "wrapper", "recompiles": 1}]
    }
    md = build_compiled_markdown(data, "full")
    assert "Recompilation" in md


def test_load_compiled_results_prefers_full(tmp_path: Path):
    full = _payload("full")
    reduced = _payload("reduced")
    (tmp_path / "compiled-full.json").write_text(json.dumps(full))
    (tmp_path / "compiled-reduced.json").write_text(json.dumps(reduced))

    data, level = load_compiled_results(tmp_path)

    assert level == "full"
    assert data["level"] == "full"


def test_load_compiled_results_falls_back_to_reduced(tmp_path: Path):
    (tmp_path / "compiled-reduced.json").write_text(json.dumps(_payload("reduced")))
    data, level = load_compiled_results(tmp_path)
    assert level == "reduced"


def test_load_compiled_results_missing(tmp_path: Path):
    assert load_compiled_results(tmp_path) == (None, None)
