"""Single-op backward checks: eager vs compiled gradients."""

from __future__ import annotations

import traceback

import torch

from runner import TOL, diff_metrics


def classify_grads(eager_grad: torch.Tensor, compiled_grad: torch.Tensor) -> dict:
    max_abs, max_rel = diff_metrics(eager_grad, compiled_grad)
    return {
        "cell_class": "fails_silent" if max_rel > TOL else "works",
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
    }


def _loss_grad(fn, x: torch.Tensor, y: torch.Tensor, compiled: bool):
    leaf = x.detach().clone().requires_grad_(True)
    target = torch.compile(fn) if compiled else fn
    out = target(leaf)
    loss = (torch.abs(out - y) ** 2).sum()
    (grad,) = torch.autograd.grad(loss, leaf, allow_unused=True)
    return None if grad is None else grad.detach()


def check_grad_agreement(fn, x: torch.Tensor, y: torch.Tensor) -> dict:
    try:
        eager_grad = _loss_grad(fn, x, y, compiled=False)
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        return {
            "status": "error",
            "cell_class": "fails_loud",
            "max_abs_diff": None,
            "max_rel_diff": None,
            "traceback_tail": traceback.format_exc().strip()[-800:],
            "notes": "raised during eager backward",
        }

    try:
        torch._dynamo.reset()
        compiled_grad = _loss_grad(fn, x, y, compiled=True)
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        return {
            "status": "error",
            "cell_class": "fails_loud",
            "max_abs_diff": None,
            "max_rel_diff": None,
            "traceback_tail": traceback.format_exc().strip()[-800:],
            "notes": "raised during compiled backward",
        }

    if eager_grad is None or compiled_grad is None:
        return {
            "status": "error",
            "cell_class": "fails_loud",
            "max_abs_diff": None,
            "max_rel_diff": None,
            "traceback_tail": None,
            "notes": "no gradient produced",
        }

    result = classify_grads(eager_grad, compiled_grad)
    result.update(status="ok", traceback_tail=None, notes="")
    return result


def run_autograd_check(case, spec, device: str, dtype: torch.dtype) -> dict:
    if spec.level == "inner" and case.inner_build is not None:
        builder = case.inner_build
    else:
        builder = case.build

    row = {
        "op": spec.op,
        "level": spec.level,
        "device": device,
        "dtype": str(dtype),
        "method": "grad",
        "config": "default",
    }
    try:
        fn, x, _ = builder(device, dtype, "fwd")
        y = torch.randn_like(fn(x).detach())
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        row.update(
            status="error",
            cell_class="fails_loud",
            traceback_tail=traceback.format_exc().strip()[-800:],
            notes="case build failed",
        )
        return row

    row.update(check_grad_agreement(fn, x, y))
    return row
