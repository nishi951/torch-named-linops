from .callbacks import (
    AbstractTrainingCallback,
    topological_sort_callbacks as sort_callbacks,
    GlobalStep,
    EpochStep,
)


@dataclass
class TrainingState:
    model: nn.Module
    loss_fn: nn.Module
    optimizer: Optional[optim.Optimizer] = None
    scheduler: Optional[optim.lr_scheduler] = None
    features: Optional[Mapping] = None
    pred: Optional[Mapping] = None
    target: Optional[Mapping] = None
    loss: Optional[torch.Tensor] = None
    global_step: int = -1
    epoch: int = -1


@dataclass
class AdamHparams:
    lr: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.9990)
    fused: bool = True
    """Used for AdamW"""


@dataclass
class MultiStepLRHparams:
    milestones: List[int]
    gamma: float


@dataclass
class BasicTrainerHparams:
    num_epochs: int
    optimizer_hparams: AdamHparams
    scheduler_hparams: Optional[MultiStepLRHparams] = None


class BasicTrainer:
    def __init__(self, hparams: BasicTrainerHparams, handlers=None):
        self.hparams = hparams
        self.handlers = defaultdict(list)

        # Default handlers
        if handlers is None:
            self.register_handler('train_step_ended', GlobalStep())
            self.register_handler('epoch_ended', EpochStep())

    def train(
        self,
        model,
        loss_fn,
        train_dataloader,
        total_train_steps: Optional[int] = None,
    ):
        s = self.initialize_training_state(model, loss_fn)
        s = self.dispatch('train_started', s)
        for epoch in tqdm(range(self.hparams.num_epochs)):
            s = self.dispatch('epoch_started', s)
            s.model.train()
            for s.features, s.target in tqdm(iter(dataloader), total=total_train_steps):
                s = self.dispatch('train_step_started', s)
                s = self.train_step(s)
                s = self.dispatch('train_step_ended', s)
            s = self.dispatch('epoch_ended', s)
        self.dispatch('train_ended', s)
        return model

    def train_step(self, s: TrainingState):
        """Override if other step behavior is desirable (e.g. gradient clipping)"""
        s.pred = s.model(s.features)
        s.loss = s.loss_fn(s.pred, s.target)
        s.optimizer.zero_grad()
        s.loss.backward()
        s = self.dispatch('backward_called', s)
        s.optimizer.step()
        s = self.dispatch('optimizer_stepped', s)
        return s

    def register_handler(self, event, handler: AbstractTrainingCallback, sort_now: bool = True):
        self.handlers[event].append(handler)
        if sort_now:
            self.handlers[event] = sort_callbacks(self.handlers[event])

    def dispatch(self, event, s: TrainingState):
        for handler in self.handlers[event]:
            s = handler(s)

    def initialize_training_state(self, model, loss_fn):
        s = TrainingState(model=model, loss_fn=loss_fn)
        s.optimizer = self.make_optimizer(model)
        s.scheduler = self.make_scheduler(s.optimizer)
        return s

    def make_optimizer(self, model):
        """Override for custom behavior"""
        return optim.Adam(
            model.parameters(),
            lr=self.hparams.optimizer_hparams.lr,
            betas=self.hparams.optimizer_hparams.betas,
        )

    def make_scheduler(self, optimizer):
        """Override for custom behavior"""
        if self.hparams.scheduler_hparams is None:
            return None
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.scheduler_hparams.milestones,
            gamma=opt.step_scheduler_hparams.gamma,
            verbose=False,
        )
        return scheduler
