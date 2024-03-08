import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def to_dev(struct, device):
    return apply_struct(
        struct,
        fn=(lambda x: x.to(device)),
        condition=lambda x: isinstance(x, torch.Tensor),
    )


@dataclass
class TrainHparams:
    dataloader_kwargs: Mapping = field(
        default_factory=lambda: {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
        }
    )
    num_epochs: int = 50


class ImplicitGROG:
    """Base class for implicit grog functionality"""

    def __init__(
        self,
        optimizer: Optional[optim.Optimizer],
        train_hparams: TrainHparams,
        scheduler: Optional = None,
    ):
        self.optimizer = (
            optimizer
            if optimizer is not None
            else optim.Adam(self.model.parameters(), lr=1e-3)
        )
        self.hparams = train_hparams
        self.scheduler = scheduler
        self.last_epoch = -1
        self.global_step = -1

    def preprocess(self, data: Mapping, device) -> Tuple[Dataset, Callable, Mapping]:
        """Should return a torch Dataset and a callable loss function"""
        raise NotImplementedError()

    def apply_model(self, data, model, device) -> Mapping:
        """ """
        raise NotImplementedError()

    def grogify(self, data, model, device) -> Mapping:
        dataset, loss_fn, data = self.preprocess(data, device)
        loss_fn.to(device)
        model.to(device)
        model = self.train(dataset, model, loss_fn, device)
        model.eval()
        gridded_data = self.apply_model(data, model, device)
        return gridded_data, model

    def train(self, dataset, model, loss_fn, device):
        dataloader = DataLoader(dataset, **self.params.dataloader_kwargs)
        batch_size = self.hparams.dataloader_kwargs["batch_size"]
        for _ in range(self.hparams.num_epochs):
            logger.info(f"Epoch {self.last_epoch + 1}")
            model.train()
            for source, target in tqdm(
                iter(dataloader), total=ceil(len(dataset) / batch_size)
            ):
                source = to_dev(source, device)
                target = to_dev(target, device)
                self.optimizer.zero_grad()
                pred = model(source)
                loss = loss_fn(pred, target)
                loss.backward()
                self.optimizer.step()
                self.global_step += 1
                self.log_train(source, target, pred, loss, logdir)
                if debug:
                    break
            self.last_epoch += 1
            if self.scheduler is not None:
                self.scheduler.step()
            if debug:
                return
        return model

    def log_train(self, source, target, pred, loss, logdir: Optional[Path] = None):
        pass

    @torch.no_grad
    def eval(self, dataset, model, device):
        kwargs = self.hparams.dataloader_kwargs
        batch_size = kwargs["batch_size"]
        dataloader = DataLoader(
            dataset,
            batch_size=kwargs["batch_size"],
            pin_memory=kwargs["pin_memory"],
            num_workers=kwargs["num_workers"],
            shuffle=False,
        )
        model.eval()
        preds = []
        for source, _ in tqdm(iter(dataloader), total=ceil(len(dataset) / batch_size)):
            source = to_dev(source, device)
            preds.append(model(source))
        return preds
