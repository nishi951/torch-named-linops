from dataclasses import dataclass
from typing import Optional, Tuple, List, Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


from utils import EventManager

from .. import AbstractTrainer


@dataclass
class TrainingState:
    model: nn.Module
    optimizer: Optional[optim.Optimizer] = None
    scheduler: Optional[optim.lr_scheduler] = None
    features: Optional[Mapping] = None
    pred: Optional[Mapping] = None
    target: Optional[Mapping] = None
    loss: Optional[torch.Tensor] = None
    global_step: int = -1
    epoch: int = -1


@dataclass
class AdamConfig:
    lr: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.9990)
    fused: bool = True
    """Used for AdamW"""


@dataclass
class MultiStepLRConfig:
    milestones: List[int]
    gamma: float


@dataclass
class TrainerConfig:
    num_epochs: int
    optimizer_config: AdamConfig
    scheduler_config: Optional[MultiStepLRConfig] = None


class Trainer(AbstractTrainer):
    def __init__(self, config: KCalTrainer, manager: Optional[EventManager] = None):
        self.config = config
        self.m = manager if manager is not None else EventManager

    def train(
        self,
        model,
        loss_fn,
        dataloader,
        total_train_steps: Optional[int] = None,
    ):
        s = self.initialize_training_state(model, loss_fn)
        s = self.dispatch("train_started", s)
        for epoch in tqdm(range(self.config.num_epochs)):
            s = self.dispatch("epoch_started", s)
            s.model.train()
            for s.features, s.target in tqdm(iter(dataloader), total=total_train_steps):
                s = self.dispatch("train_step_started", s)
                s = self.train_step(s, loss_fn)
                s = self.dispatch("train_step_ended", s)
            s = self.dispatch("epoch_ended", s)
        self.dispatch("train_ended", s)
        return model

    def train_step(self, s: TrainingState, loss_fn):
        """Override if other step behavior is desirable (e.g. gradient clipping)"""
        s.pred = s.model(s.features)
        s.loss = loss_fn(s.pred, s.target)
        s.optimizer.zero_grad()
        s.loss.backward()
        s = self.dispatch("backward_called", s)
        s.optimizer.step()
        s = self.dispatch("optimizer_stepped", s)
        return s

    def initialize_training_state(self, model):
        s = TrainingState(model=model)
        s.optimizer = self.build_optimizer(model)
        s.scheduler = self.build_scheduler(s.optimizer)
        return s

    def build_optimizer(self, model):
        """Override for custom behavior"""
        return optim.Adam(
            model.parameters(),
            lr=self.config.optimizer_config.lr,
            betas=self.config.optimizer_config.betas,
        )

    def build_scheduler(self, optimizer):
        """Override for custom behavior"""
        if self.config.scheduler_config is None:
            return None
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.scheduler_config.milestones,
            gamma=self.config.scheduler_config.gamma,
            verbose=False,
        )
        return scheduler
