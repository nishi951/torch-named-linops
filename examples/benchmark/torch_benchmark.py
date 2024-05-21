import copy
import gc
import logging

import torch
from easydict import EasyDict

logger = logging.getLogger(__name__)

__all__ = ['TorchHandler']


class TorchHandler:
    def __init__(self):
        self.reset()

    def reset(self):
        self._start_event = None
        self._end_event = None

        self.result = EasyDict(
            {
                'timings_ms': [],
                'max_mem_bytes': None,
            }
        )

    def bench_start(self, *args, **kwargs):
        self.reset()
        gc.disable()
        torch.cuda.reset_peak_memory_stats()

    def bench_end(self, *args, **kwargs):
        self.result.max_mem_bytes = torch.cuda.max_memory_allocated()
        gc.enable()
        logger.info(f'Max memory allocated: {self.result.max_mem_bytes}')

    def trial_start(self, event, i):
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def trial_end(self, event, i):
        self._end_event.record()
        torch.cuda.synchronize()
        time = self._start_event.elapsed_time(self._end_event)
        logger.info(f'Trial {i}: {time}')
        self.result.timings_ms.append(time)

    def collect_results(self, event, data):
        return {'torch': copy.deepcopy(self.result)}

