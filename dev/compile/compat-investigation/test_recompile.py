"""Plain-assert verification for shape-driven recompilation checks."""

import torch

from recompile import measure_recompiles


def test_two_shapes_recompile():
    torch.manual_seed(0)
    W = torch.randn(7, 3, device="cuda")
    inputs = [
        torch.randn(4, 7, device="cuda"),
        torch.randn(8, 7, device="cuda"),
    ]

    count, outs = measure_recompiles(lambda x: x @ W, inputs)

    assert count >= 1, count
    for out, x in zip(outs, inputs):
        assert torch.allclose(out, x @ W, atol=1e-5), "output drifted"


def test_recompile_count_is_per_measurement():
    torch.manual_seed(0)
    W = torch.randn(7, 3, device="cuda")
    inputs = [
        torch.randn(4, 7, device="cuda"),
        torch.randn(8, 7, device="cuda"),
    ]

    first, _ = measure_recompiles(lambda x: x @ W, inputs)
    second, _ = measure_recompiles(lambda x: x @ W, inputs)

    assert first == second >= 1, (first, second)


if __name__ == "__main__":
    test_two_shapes_recompile()
    print("test_two_shapes_recompile PASS")
    test_recompile_count_is_per_measurement()
    print("test_recompile_count_is_per_measurement PASS")
