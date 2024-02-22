from dataclasses import dataclass, field, asdict
from math import ceil
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    pass
from typing import Optional, Mapping, Callable
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchlinops.utils import apply_struct

logger = logging.getLogger(__name__)

__all__ = [
    'TrainHparams',
    'Trainer',
]

def unbatch_and_to_dev(struct, device):
    return apply_struct(
        struct,
        fn=lambda x: x[0].to(device),
        condition=lambda x: isinstance(x, torch.Tensor)
    )


@dataclass
class TrainHparams:
    dataloader_kwargs: Mapping = field(default_factory = lambda: {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True,
    })
    train_log_every: int = 1
    """Number of training steps to log after"""
    checkpoint_every: int = 100
    """Number of epochs after which to keep checkpoints
    TODO Always keep latest checkpoint?"""


@dataclass
class TrainCheckpoint:
    model_ckpt: Mapping
    train_hparams: TrainHparams
    optimizer_ckpt: Mapping
    scheduler_ckpt: Optional[Mapping]
    last_epoch: Optional[int]
    global_step: int


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            loss_fn: Callable,
            train_hparams: TrainHparams,
            scheduler: Optional = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.hparams = train_hparams
        self.last_epoch = -1
        self.global_step = -1

    def train(
            self,
            dataset,
            num_epochs: int,
            val_dataset: Optional[Dataset] = None,
            val_fn: Callable = None,
            logdir: Optional[Path] = None,
            debug: bool = False,
    ):
        dataloader = DataLoader(dataset, **self.hparams.dataloader_kwargs)
        batch_size = self.hparams.dataloader_kwargs['batch_size']
        for _ in range(num_epochs):
            logger.info(f'Epoch {self.last_epoch + 1}')
            self.model.train()
            for source, target in tqdm(iter(dataloader), total=ceil(len(dataset)/batch_size),):
                # Assume batch size is always 1
                # TODO account for multiple devices
                source = unbatch_and_to_dev(source, self.model.start_device)
                target = unbatch_and_to_dev(target, self.model.start_device)
                self.optimizer.zero_grad()
                pred = self.model(source)
                loss = self.loss_fn(pred, target)
                loss.backward()
                self.optimizer.step()
                self.global_step += 1
                self.log_train(source, target, pred, loss, logdir)
                if debug:
                    break
            self.last_epoch += 1
            if val_dataset is not None and val_fn is not None:
                self.val(val_dataset, val_fn, logdir, debug)
            if self.scheduler is not None:
                self.scheduler.step()
            if debug:
                return

    @torch.no_grad()
    def val(self, dataset, eval_fn, logdir, debug: bool = False):
        self.model.eval()
        dataloader = DataLoader(dataset, **self.hparams.dataloader_kwargs)
        val_result = {}

        # Validation Loss
        val_loss = 0.
        batch_size = self.hparams.dataloader_kwargs['batch_size']
        for source, target in tqdm(iter(dataloader), total=ceil(len(dataset)/batch_size)):
            source = unbatch_and_to_dev(source, self.model.start_device)
            target = unbatch_and_to_dev(target, self.model.start_device)
            pred = self.model.val_forward(source)
            val_loss += eval_fn(pred, target)
            if debug:
                break
        val_loss /= len(dataset)
        val_result['loss'] = val_loss

        # TODO: Add other validation metrics

        self.log_val(source, target, pred, val_result, logdir)
        self.manage_checkpoints(
            logdir/'checkpoints', self.last_epoch, self.hparams.checkpoint_every
        )

    def log_train(self, source, target, pred, loss, logdir: Optional[Path] = None):
        return NotImplemented

    def log_val(self, source, target, pred, val_result, logdir: Optional[Path] = None):
        return NotImplemented

    def manage_checkpoints(self, ckpt_dir, last_epoch, checkpoint_every):
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        # Update latest checkpoint
        ckpt_latest = ckpt_dir/'latest.pt'
        ckpt_latest_bak = ckpt_dir/'latest.pt.bak'
        if (ckpt_dir/'latest.pt').is_file():
            # Backup
            ckpt_latest.rename(ckpt_latest_bak)
        # Save new checkpoint
        self.save(ckpt_latest)
        if ckpt_latest_bak.is_file():
            # Remove backup
            ckpt_latest_bak.unlink()
        if not (last_epoch % checkpoint_every):
            # Also save epoch checkpoint
            self.save(ckpt_dir/f'epoch_{last_epoch:06d}.pt')


    def save(self, logfile: Path):
        logging.info(f'Saving checkpoint to {logfile}')
        ckpt = TrainCheckpoint(
            model_ckpt=self.model.state_dict(),
            train_hparams=self.hparams,
            optimizer_ckpt=self.optimizer.state_dict(),
            scheduler_ckpt=self.scheduler.state_dict() if self.scheduler is not None else None,
            last_epoch=self.last_epoch,
            global_step=self.global_step,
        )
        torch.save(asdict(ckpt), logfile)
        return ckpt

    def load(self, ckpt_path: Path):
        """Restarting training"""
        ckpt = TrainCheckpoint(**torch.load(ckpt_path))
        self.model.load_state_dict(torch.load(ckpt.model_ckpt))
        self.optimizer.load_state_dict(torch.load(ckpt.optimizer_ckpt))
        if ckpt.scheduler_ckpt is not None:
            self.scheduler.load_state_dict(torch.load(ckpt.scheduler_ckpt))

        with open(ckpt.train_hparams, 'r') as f:
            hparams = yaml.full_load(f)
        self.train_hparams = TrainHparams(**hparams)

        self.last_epoch = ckpt.last_epoch
        return self
