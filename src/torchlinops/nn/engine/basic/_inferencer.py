from dataclasses import dataclass
from typing import Optional, Mapping

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import EventManager
from .. import AbstractInferencer

__all__ = [
    "InferenceState",
    "BasicInferencer",
]


@dataclass
class InferenceState:
    model: nn.Module
    features: Optional[Mapping] = None
    pred: Optional[Mapping] = None
    target: Optional[Mapping] = None
    metrics: Optional[Mapping] = None
    global_step: int = -1
    epoch: int = -1


class Inferencer(AbstractInferencer):
    """Apply a trained model to held-out data

    Validation and Testing are combined under "Inference" in that in both
    cases, the model itself should not glean any information from the data applied.
    However, future models may be trained differently depending on the results of validation

    Testing should only be performed once, typically at the very end.
    """

    def __init__(self, manager: Optional[EventManager] = None):
        self.m = manager if manager is not None else EventManager

    def val(self, model: nn.Module, val_dataloader):
        """Run a validation

        Validation occurs during the model iteration process. While it is like
        testing in that the model is applied to unseen data, it is
        distinct from testing in that the model's hyperparameters can be updated
        based on results from validation.
        """
        s = self.initialize_inference_state(model)
        self.m.dispatch("val_started", s)
        for s.features, s.target in tqdm(iter(val_dataloader), "Validation"):
            self.m.dispatch("val_step_started", s)
            s.pred = self.infer(s.model, s.features)
            self.m.dispatch("val_step_ended", s)
        self.m.dispatch("val_ended", s)
        return s

    def test(self, model: nn.Module, test_dataloader):
        """Run a test"""
        s = self.initialize_inference_state(model)
        self.m.dispatch("test_started", s)
        for features, _ in tqdm(iter(test_dataloader), "Testing"):
            self.m.dispatch("test_step_started", s)
            s.pred = self.infer(s.model, s.features)
            self.m.dispatch("test_step_ended", s)
        self.m.dispatch("test_ended", s)
        return s

    def initialize_inference_state(model):
        model.eval()
        s = InferenceState(model)
        return s

    @torch.no_grad()
    def infer(self, model, features):
        """Convenience function"""
        return model(features)
