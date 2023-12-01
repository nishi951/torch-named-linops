import functools

import tyro

from .log import setup_console_logger


class Experiment:
    """Context manager to measure how much time was spent in the target scope."""

    def __init__(self, config_cls, main_fn):
        self.config_cls = config_cls
        self.main_fn = main_fn

    def run(self):
        opt = tyro.cli(self.config_cls)
        logger = logging.getLogger()
        if logger is not None:
            setup_console_logger(logger, log_level)
        ret = self.main_fn(opt)
        return ret
