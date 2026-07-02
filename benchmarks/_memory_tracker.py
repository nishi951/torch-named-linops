"""Thin wrapper around torch.utils.benchmark that adds GPU memory tracking."""

import torch
import torch.utils.benchmark as benchmark


def benchmark_op(fn, device="cpu", min_run_time=None, num_threads=None):
    """Benchmark a callable using torch.utils.benchmark with peak memory tracking.

    Parameters
    ----------
    fn : callable
        Function to benchmark, must take no arguments.
    device : str, optional
        "cpu" or "cuda". Used to decide min_run_time and memory tracking.
    min_run_time : float, optional
        Minimum total run time passed to blocked_autorange.
        Defaults to 0.05 for cuda and 0.2 for cpu.
    num_threads : int, optional
        Number of threads for the benchmark. Defaults to torch.get_num_threads().

    Returns
    -------
    measurement : torch.utils.benchmark.Measurement
        Timing measurement from blocked_autorange().
    peak_mem_bytes : int or None
        Peak GPU memory allocated during the benchmark, or None for cpu.
    """
    if min_run_time is None:
        min_run_time = 0.05 if device == "cuda" else 0.2

    if num_threads is None:
        num_threads = torch.get_num_threads()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        num_threads=num_threads,
    )
    measurement = timer.blocked_autorange(min_run_time=min_run_time)

    peak_mem = None
    if device == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()

    return measurement, peak_mem
