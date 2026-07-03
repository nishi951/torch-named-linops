"""Cupy-native blocked_autorange benchmark, mirroring torch.utils.benchmark."""

import numpy as np


class CupyMeasurement:
    """Mimics torch.utils.benchmark.Measurement interface.

    Attributes
    ----------
    times : list of float
        Per-run times in seconds, one entry per measurement block.
    number_per_run : int
        Number of function executions per measurement block.
    mean : float
        Mean time per run in seconds.
    median : float
        Median time per run in seconds.
    iqr : float
        Interquartile range of per-run times in seconds.
    peak_mem_bytes : int or None
        Peak GPU memory used during the benchmark, in bytes.
    """

    def __init__(self, times, number_per_run, peak_mem_bytes=None):
        self.times = times
        self.number_per_run = number_per_run
        self.peak_mem_bytes = peak_mem_bytes
        self.mean = float(np.mean(times)) if times else 0.0
        self.median = float(np.median(times)) if times else 0.0
        if len(times) > 1:
            q75, q25 = np.percentile(times, [75, 25])
            self.iqr = float(q75 - q25)
        else:
            self.iqr = 0.0


def cupy_blocked_autorange(fn, min_run_time=0.05, min_runs=10):
    """blocked_autorange-style benchmark for cupy/sigpy GPU functions.

    Mirrors ``torch.utils.benchmark.Timer.blocked_autorange()`` but uses
    ``cp.cuda.Event`` for timing and the cupy memory pool for memory tracking.

    Parameters
    ----------
    fn : callable
        Function to benchmark (no arguments).
    min_run_time : float
        Minimum total run time in seconds.
    min_runs : int
        Minimum number of runs to perform.

    Returns
    -------
    CupyMeasurement
        Measurement object with timing and memory statistics.
    """
    import cupy as cp

    # Warmup
    fn()
    cp.cuda.runtime.deviceSynchronize()

    # Auto-calibrate runs_per_block: start at 1, double until block > 5ms
    runs_per_block = 1
    while True:
        start = cp.cuda.Event(disable_timing=False)
        end = cp.cuda.Event(disable_timing=False)
        start.record()
        for _ in range(runs_per_block):
            fn()
        end.record()
        end.synchronize()
        block_time_ms = cp.cuda.get_elapsed_time(start, end)
        if block_time_ms > 5.0:
            break
        runs_per_block *= 2
        if runs_per_block > 1000000:
            break

    # Reset memory pool for clean measurement
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    peak_mem = 0

    # Collect measurements until min_run_time is reached AND we have min_runs
    total_time_s = 0.0
    timings_s = []
    while total_time_s < min_run_time or len(timings_s) < min_runs:
        start = cp.cuda.Event(disable_timing=False)
        end = cp.cuda.Event(disable_timing=False)
        start.record()
        for _ in range(runs_per_block):
            fn()
        end.record()
        end.synchronize()
        block_time_ms = cp.cuda.get_elapsed_time(start, end)
        per_run_s = (block_time_ms / runs_per_block) / 1000.0
        timings_s.append(per_run_s)
        total_time_s += block_time_ms / 1000.0

        # Track peak memory
        current_mem = mempool.total_bytes()
        if current_mem > peak_mem:
            peak_mem = current_mem

    return CupyMeasurement(timings_s, runs_per_block, peak_mem)
