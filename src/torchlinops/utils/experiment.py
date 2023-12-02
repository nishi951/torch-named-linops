from dataclasses import fields
import logging

import tyro

from .log import setup_console_logger

__all__ = ['Experiment']

class Experiment:
    """Context manager to measure how much time was spent in the target scope."""

    def __init__(self, config_cls, main_fn):
        # Make sure log level is a field
        assert 'log_level' in [f.name for f in fields(config_cls)], \
            'Experiment config must include a "log_level" field for logging.'
        self.config_cls = config_cls
        self.main_fn = main_fn

    def run(self):
        opt = tyro.cli(self.config_cls)
        logger = logging.getLogger()
        if logger is not None:
            setup_console_logger(logger, opt.log_level)
        ret = self.main_fn(opt)
        return ret
