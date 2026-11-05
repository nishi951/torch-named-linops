import torch

from benchmarks._compiled_runner import (
    TOL,
    CellSpec,
    capture_graph_breaks,
    check_grad_agreement,
    detect_degradation,
    diff_metrics,
    effective_class,
    execute_cell,
    measure_recompiles,
    median_time,
)


def _spec(config="default", device="cpu"):
    return CellSpec(
        op="op",
        category="linop",
        method="fwd",
        device=device,
        dtype=torch.float32,
        level="wrapper",
        config=config,
    )


def test_diff_metrics_exact_match_is_zero():
    x = torch.randn(4)
    assert diff_metrics(x, x.clone()) == (0.0, 0.0)


def test_diff_metrics_shape_mismatch_is_inf():
    assert diff_metrics(torch.randn(2), torch.randn(1, 2)) == (
        float("inf"),
        float("inf"),
    )


def test_diff_metrics_nan_is_inf():
    ref = torch.tensor([1.0, 2.0])
    out = torch.tensor([1.0, float("nan")])
    assert diff_metrics(ref, out) == (float("inf"), float("inf"))


def test_nan_output_is_fails_silent():
    from benchmarks._compiled_runner import run_cell

    res = run_cell(_spec(), torch.tensor([1.0, 2.0]), torch.tensor([1.0, float("nan")]))
    assert res.cell_class == "fails_silent"


def test_shape_disagreement_is_fails_silent():
    from benchmarks._compiled_runner import run_cell

    ref = torch.randn(2)
    res = run_cell(_spec(), ref, ref.reshape(1, 2))
    assert res.cell_class == "fails_silent"


def test_effective_class_taxonomy():
    assert effective_class("ok", "works", 0, 1.2, False) == "works"
    assert effective_class("ok", "works", 2, 1.2, False) == "works_with_breaks"
    assert effective_class("ok", "works", 0, 0.5, False) == "compiles_slower"
    assert effective_class("ok", "works", 0, 1.2, True) == "silent_degradation"
    assert effective_class("ok", "fails_silent", 0, None, False) == "fails_silent"
    assert effective_class("n_a", "n_a", 0, None, False) == "n_a"
    assert (
        effective_class("ok", "works", None, 1.2, False, "default")
        == "works_with_unknown_breaks"
    )
    assert effective_class("ok", "works", None, 1.2, False, "fullgraph") == "works"


def test_detect_degradation():
    assert detect_degradation(["falling back to eager"]) is True
    assert detect_degradation([]) is False


def test_median_time_counts_warmup_and_iters():
    from benchmarks._compiled_runner import CallCounter

    counter = CallCounter()
    median_time(
        lambda: torch.ones(2) + 1, warmup=3, iters=5, device="cpu", counter=counter
    )
    assert counter.count >= 8


def test_execute_cell_works_on_cpu():
    weight = torch.randn(7, 3)
    x = torch.randn(7)
    res = execute_cell(_spec(), lambda t: t @ weight, x, warmup=2, iters=3)
    assert res.status == "ok", res.traceback_tail
    assert res.cell_class == "works"
    assert res.eager_time > 0 and res.compiled_time > 0
    assert isinstance(res.graph_breaks, int)


def test_execute_cell_reports_failure():
    def target(t):
        raise RuntimeError("deliberate")

    res = execute_cell(_spec(), target, torch.randn(3), warmup=1, iters=1)
    assert res.status == "error"
    assert res.cell_class == "fails_loud"


def test_capture_graph_breaks_records_explain_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("explain broke")

    monkeypatch.setattr(torch._dynamo, "explain", boom)
    count, locations = capture_graph_breaks(lambda t: t, torch.ones(2), "boom")
    assert count is None
    assert any("explain failed" in loc for loc in locations)


def test_capture_graph_breaks_includes_user_stack():
    def target(t):
        y = t + 1
        torch._dynamo.graph_break()
        return y * 2

    count, locations = capture_graph_breaks(target, torch.ones(4), "stack")

    assert count >= 1
    assert any("test_compiled_runner.py" in location for location in locations)


def test_execute_cell_records_compiled_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("compiled boom")

    monkeypatch.setattr(torch, "compile", boom)
    res = execute_cell(_spec(), lambda t: t * 2, torch.randn(3), warmup=1, iters=1)
    assert res.status == "error"
    assert res.cell_class == "fails_loud"
    assert "compiled boom" in (res.traceback_tail or "")


def test_fullgraph_rows_have_unknown_breaks():
    weight = torch.randn(7, 3)
    x = torch.randn(7)
    res = execute_cell(
        _spec(config="fullgraph"),
        lambda t: t @ weight,
        x,
        warmup=1,
        iters=1,
    )
    assert res.status == "ok", res.traceback_tail
    assert res.graph_breaks is None


def test_measure_recompiles_two_shapes():
    weight = torch.randn(7, 3)
    inputs = [torch.randn(4, 7), torch.randn(8, 7)]
    count, outputs = measure_recompiles(lambda x: x @ weight, inputs)
    assert count >= 1
    for out, x in zip(outputs, inputs):
        torch.testing.assert_close(out, x @ weight)


def test_check_grad_agreement_cpu():
    def fn(t):
        return t * 2.0

    result = check_grad_agreement(fn, torch.randn(3), torch.randn(3))
    assert result["cell_class"] == "works"
