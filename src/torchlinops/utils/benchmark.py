import time
import gc
import logging

import torch

logger = logging.getLogger(__name__)

__all__ = [
    'torch_benchmark',
    'Timer',
]

def torch_benchmark(fn, n_trials, warmup=True, *args, **kwargs):
    """Run a function n times with specified args and kwargs and report the time."""
    if warmup:
        fn(*args, **kwargs)
    timings_ms = []
    torch.cuda.reset_peak_memory_stats()
    gc.disable()
    for i in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn(*args, **kwargs)
        end.record()

        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        logger.info(f'Trial {i}: {time}')

        timings_ms.append(time)
    gc.enable()
    max_mem_bytes = torch.cuda.max_memory_allocated()
    logger.info(f'Max memory allocated: {max_mem_bytes}')
    return timings_ms, max_mem_bytes

class Timer:
    """Context manager to measure how much time was spent in the target scope."""

    def __init__(self, name: str, func=time.perf_counter, log_level: int = None):
        self.name = name
        self._func = func
        self.start = None
        self.total = None
        self.log_level = log_level if log_level is not None else logging.NOTSET

    def __enter__(self):
        self.start = self._func()

    def __exit__(self, type=None, value=None, traceback=None):
        self.total = (self._func() - self.start) # Store the desired value.
        logging.log(level=self.log_level, msg=f'{self.name}: {self.total:0.5f} s')
