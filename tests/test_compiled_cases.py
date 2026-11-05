import json
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks._compiled_cases import (
    CASES,
    SKIPPED,
    configs_for,
    iter_specs,
    levels_for,
)
from benchmarks._compiled_runner import CellSpec
from benchmarks.conftest import (
    DIFF_PATHS,
    _write_compiled_results,
    remove_owned_latest_files,
)


def test_registry_covers_all_linops_and_functional_kernels():
    ops = {case.op for case in CASES}
    assert {"Dense", "Chain", "NUFFT", "Interpolate", "ArrayToBlocks"} <= ops
    assert {
        "functional.unfold",
        "functional.fold",
        "functional.grid",
        "functional.ungrid",
    } <= ops
    assert "BreakpointLinop" in SKIPPED


def test_full_methods_include_normal_and_all_dtypes():
    dense = next(case for case in CASES if case.op == "Dense")
    assert dense.methods == ["fwd", "adj", "normal"]
    assert dense.dtypes == [torch.float32]


def test_deep_dive_cases_have_inner_build():
    nufft = next(case for case in CASES if case.op == "NUFFT")
    assert nufft.deep_dive is True
    assert nufft.inner_build is not None
    assert levels_for(nufft, "reduced") == ["wrapper", "inner"]


def test_reduced_specs_drop_normal_and_fullgraph():
    specs = iter_specs("reduced")
    assert {spec.method for spec in specs} == {"fwd", "adj"}
    assert {spec.config for spec in specs} == {"default"}
    assert all(spec.device in {"cpu", "cuda"} for spec in specs)


def test_full_specs_include_normal_and_fullgraph():
    specs = iter_specs("full")
    assert {spec.method for spec in specs} == {"fwd", "adj", "normal"}
    assert {spec.config for spec in specs} == {"default", "fullgraph"}
    assert configs_for("full") == ["default", "fullgraph"]


def test_functional_cases_are_inner_only():
    specs = [s for s in iter_specs("reduced") if s.category == "functional"]
    assert specs
    assert {s.level for s in specs} == {"inner"}


def test_case_by_op_includes_functional_kernels():
    from benchmarks._compiled_cases import CASE_BY_OP

    assert "functional.unfold" in CASE_BY_OP


def test_compiled_session_preserves_eager_files(tmp_path: Path):
    (tmp_path / "results.json").write_text("eager")
    (tmp_path / "compiled-reduced.json").write_text("old")

    remove_owned_latest_files(tmp_path, ["compiled-reduced.json"])

    assert (tmp_path / "results.json").read_text() == "eager"
    assert not (tmp_path / "compiled-reduced.json").exists()


def test_entry_records_builder_failure(monkeypatch):
    from benchmarks import test_compiled_benchmarks as entry

    class BoomCase:
        op = "Boom"
        category = "linop"
        deep_dive = False
        inner_build = None

        def build(self, *args):
            raise RuntimeError("build boom")

    class Session:
        def __init__(self):
            self.recorded = []

        def record(self, result):
            self.recorded.append(result)

    spec = CellSpec(
        op="Boom",
        category="linop",
        method="fwd",
        device="cpu",
        dtype=torch.float32,
        level="wrapper",
        config="default",
    )
    monkeypatch.setitem(entry.CASE_BY_OP, "Boom", BoomCase())
    session = Session()

    entry.run_compiled_row(spec, session)

    assert len(session.recorded) == 1
    result = session.recorded[0]
    assert result.status == "error"
    assert result.cell_class == "fails_loud"
    assert "build boom" in (result.traceback_tail or "")


def test_phase_only_results_are_written(tmp_path: Path):
    config = SimpleNamespace(
        _compiled_level="full",
        _compiled_results=[],
        _compiled_phases={"recompile": [{"op": "Dense", "recompiles": 1}]},
    )

    written = _write_compiled_results(config, tmp_path)

    assert written is True
    payload = json.loads((tmp_path / "compiled-full.json").read_text())
    assert payload["rows"] == []
    assert payload["phases"]["recompile"][0]["op"] == "Dense"


def test_dirty_paths_include_benchmarks():
    assert "benchmarks/" in DIFF_PATHS
