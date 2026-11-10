import pytest
import torch

from torchlinops import Identity
from torchlinops.tools import compile_check


def _check(linop=None, **kwargs):
    linop = linop or Identity(ishape=("N",))
    x = torch.randn(16)
    kwargs.setdefault("explain_graph_breaks", False)
    kwargs.setdefault("print_results", False)
    return compile_check(linop, x, warmup=1, iters=2, **kwargs)


def test_compile_check_defaults_to_forward_and_adjoint():
    result = _check()

    assert [row.variant for row in result.rows] == ["forward", "adjoint"]
    assert all(row.status == "ok" for row in result.rows)
    assert result.input_shape == (16,)
    assert result.output_shape == (16,)
    assert result.input_bytes == 16 * torch.tensor(0.0).element_size()
    assert all(row.speedup is not None for row in result.rows)


def test_compile_check_supports_normal_and_aliases():
    result = _check(variants=["fwd", "adj", "normal"])

    assert [row.variant for row in result.rows] == [
        "forward",
        "adjoint",
        "normal",
    ]
    assert all(row.status == "ok" for row in result.rows)


def test_compile_check_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        _check(variants=["backward"])


def test_compile_check_rejects_non_tensor_input():
    with pytest.raises(TypeError, match="sample_input"):
        compile_check(Identity(ishape=("N",)), [1, 2, 3])


def test_compile_check_prints_summary(capsys):
    _check(variants=["forward"], print_results=True)
    captured = capsys.readouterr()
    assert "compile_check:" in captured.out
    assert "forward" in captured.out


def test_compile_check_eager_failure_raises():
    class Boom:
        def __call__(self, x):
            raise RuntimeError("eager boom")

    with pytest.raises(RuntimeError, match="could not run"):
        compile_check(Boom(), torch.randn(3), print_results=False)


def test_compile_check_records_compiled_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("compiled boom")

    monkeypatch.setattr(torch, "compile", boom)
    result = _check(variants=["forward"])

    assert result.rows[0].status == "error"
    assert "compiled boom" in result.rows[0].error


def test_compile_check_fullgraph_leaves_breaks_unknown():
    result = _check(variants=["forward"], fullgraph=True, explain_graph_breaks=True)

    assert result.rows[0].status == "ok"
    assert result.rows[0].graph_breaks is None


def test_compile_check_records_graph_breaks(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("explain boom")

    monkeypatch.setattr(torch._dynamo, "explain", boom)
    result = _check(variants=["forward"], explain_graph_breaks=True)

    assert result.rows[0].graph_breaks is None
    assert any(
        "explain failed" in location for location in result.rows[0].break_locations
    )


def test_compile_check_complex_input():
    x = torch.randn(16, dtype=torch.complex64)
    result = compile_check(
        Identity(ishape=("N",)),
        x,
        variants=["forward", "adjoint"],
        warmup=1,
        iters=2,
        explain_graph_breaks=False,
        print_results=False,
    )

    assert result.dtype == torch.complex64
    assert all(row.status == "ok" for row in result.rows)


def test_compile_check_deduplicates_variants():
    result = _check(variants=["forward", "fwd"])

    assert [row.variant for row in result.rows] == ["forward"]


def test_diff_metrics_flags_shape_and_nonfinite():
    from torchlinops.tools import _diff_metrics

    assert _diff_metrics(torch.randn(2), torch.randn(1, 2)) == (
        float("inf"),
        float("inf"),
    )
    assert _diff_metrics(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, float("nan")])
    ) == (float("inf"), float("inf"))
