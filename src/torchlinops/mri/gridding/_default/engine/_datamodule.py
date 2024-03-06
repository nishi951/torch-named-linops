from torch.utils.data import DataLoader

__all__ = ['AbstractDataModule']


class AbstractDataModule:
    """A collection of data for training and evaluating models."""

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError()
