import gc
import logging

import torch

logger = logging.getLogger(__name__)

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
