import pytest
import torch

from torchlinops import Identity
from torchlinops.testing import compile_check


def _check(linop=None, **kwargs):
    linop = linop or Identity(ishape=("N",))
    x = torch.randn(16)
    return compile_check(
        linop,
        x,
        warmup=1,
        iters=2,
        explain_graph_breaks=False,
        print_results=False,
        **kwargs,
    )


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
