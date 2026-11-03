"""Plain-assert verification for single-op autograd checks."""

import torch

from autograd_check import check_grad_agreement, classify_grads


def test_grad_agreement_matmul():
    torch.manual_seed(0)
    W = torch.randn(3, 7, device="cuda")
    x = torch.randn(7, device="cuda")
    y = torch.randn(3, device="cuda")

    result = check_grad_agreement(lambda t: W @ t, x, y)

    assert result["status"] == "ok", result.get("traceback_tail")
    assert result["cell_class"] == "works", result["cell_class"]
    assert result["max_rel_diff"] < 1e-3, result["max_rel_diff"]


def test_corrupted_gradient_is_fails_silent():
    eager = torch.randn(7, device="cuda")
    compiled = eager * 1.01

    result = classify_grads(eager, compiled)

    assert result["cell_class"] == "fails_silent", result["cell_class"]


if __name__ == "__main__":
    test_grad_agreement_matmul()
    print("test_grad_agreement_matmul PASS")
    test_corrupted_gradient_is_fails_silent()
    print("test_corrupted_gradient_is_fails_silent PASS")
