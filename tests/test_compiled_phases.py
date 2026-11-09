import torch

from benchmarks.compile.phases import pipeline_break_kind


def test_pipeline_break_kind_attributes_wrapper_logging():
    assert (
        pipeline_break_kind(["at SyncContext logger.debug"])
        == "wrapper (logging/namedlinop)"
    )


def test_pipeline_break_kind_attributes_triton():
    assert (
        pipeline_break_kind(["triton kernel unsupported"]) == "inner (triton/inductor)"
    )


def test_pipeline_break_kind_none():
    assert pipeline_break_kind([]) == "none"


def test_simple_chain_builds_on_cpu():
    from benchmarks.compile.phases import _simple_chain

    chain, x = _simple_chain("cpu")
    assert chain(x).shape == (2, 8)


def test_pipeline_row_default_breaks_are_unknown():
    from benchmarks.compile.phases import _row

    row = _row("pipeline.x", "loop", "eager", torch.float32)
    assert row["graph_breaks"] is None


def test_pipeline_phase_records_chain_build_failure(monkeypatch):
    from benchmarks.compile import phases as _compiled_phases

    def boom(device):
        raise RuntimeError("chain build boom")

    monkeypatch.setattr(_compiled_phases, "CHAINS", {"simple": boom})

    rows = _compiled_phases.run_pipeline_phase(device="cpu")

    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["cell_class"] == "fails_loud"
    assert "chain build boom" in (rows[0]["traceback_tail"] or "")


def test_autograd_check_builds_forward_target(monkeypatch):
    from benchmarks.compile import phases
    from benchmarks.compile.cases import CASE_BY_OP
    from benchmarks.compile.runner import CellSpec

    called = {}

    def fake_check(fn, x, y):
        called["ok"] = True
        return {"status": "ok", "cell_class": "works"}

    monkeypatch.setattr(phases, "check_grad_agreement", fake_check)
    spec = CellSpec(
        op="Dense",
        category="linop",
        method="grad",
        device="cpu",
        dtype=torch.float32,
        level="wrapper",
        config="default",
    )

    row = phases.run_autograd_check(CASE_BY_OP["Dense"], spec, "cpu", torch.float32)

    assert row.get("notes") != "case build failed"
    assert called.get("ok") is True
