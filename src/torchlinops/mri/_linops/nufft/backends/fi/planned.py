from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd import Function


__all__ = [
    "FiNUFFTCombinedPlan",
    "PlannedFiNUFFT",
    "PlannedAdjointFiNUFFT",
    "nufft",
    "nufft_adjoint",
]


@dataclass
class FiNUFFTCombinedPlan:
    _forward: Any
    _adjoint: Any
    plan_type: str = "cpu"

    def execute(self, x: torch.Tensor, out=None) -> torch.Tensor:
        if self.plan_type == "cpu":
            x = x.detach().cpu().numpy()
        y = self._forward.execute(x, out)
        if out is None:
            out = y
        if self.plan_type == "cpu":
            out = torch.from_numpy(out)
        return out

    def adj_execute(self, x: torch.Tensor, out=None):
        if self.plan_type == "cpu":
            x = x.detach().cpu().numpy()
        y = self._adjoint.execute(x, out)
        if out is None:
            out = y
        if self.plan_type == "cpu":
            out = torch.from_numpy(out)
        return out


class PlannedFiNUFFT(Function):
    """Forward (type-2) NUFFT with planning
    Uses an external plan to keep track
    of coordinates and other parameters.
    """

    @staticmethod
    def forward(
        input: torch.Tensor,
        plan: FiNUFFTCombinedPlan,
        out: torch.Tensor,
    ):
        return plan.execute(input, out)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, plan, out = inputs
        ctx.plan = plan

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_plan = grad_out = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.plan.adj_execute(grad_output)
        return grad_input, grad_plan, grad_out


def nufft(input, plan, out=None):
    return PlannedFiNUFFT.apply(input, plan, out)


class PlannedAdjointFiNUFFT(Function):
    """Adjoint (type-1) NUFFT with planning
    Uses an external plan to keep track
    of coordinates and other parameters.
    """

    @staticmethod
    def forward(
        input: torch.Tensor,
        plan: FiNUFFTCombinedPlan,
        out: torch.Tensor,
    ):
        return plan.adj_execute(input, out)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, plan, out = inputs
        ctx.plan = plan

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_plan = grad_out = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.plan.execute(grad_output)
        return grad_input, grad_plan, grad_out


def nufft_adjoint(input, plan, out=None):
    return PlannedAdjointFiNUFFT.apply(input, plan, out)
