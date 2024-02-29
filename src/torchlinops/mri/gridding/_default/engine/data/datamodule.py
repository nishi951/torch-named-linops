
__all__ = ['AbstractDataModule']

class AbstractDataModule:
    def __init__(self, hparams):
        self.hparams = hparams

    def train_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        raise NotImplementedError()
