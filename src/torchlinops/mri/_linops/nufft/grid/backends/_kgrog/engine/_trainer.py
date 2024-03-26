__all__ = ["AbstractTrainer"]


class AbstractTrainer:
    """Base class for trainer"""

    def train(self):
        raise NotImplementedError()
