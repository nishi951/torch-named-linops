"""Thin wrapper around torch.utils.benchmark that adds GPU memory tracking."""

import time
import torch
import torch.utils.benchmark as benchmark


def benchmark_op(
    fn, device="cpu", min_run_time=None, num_threads=None, min_runs=10, data_gen_fn=None
):
    """Benchmark a callable using manual timing with optional separate data generation.

    Parameters
    ----------
    fn : callable
        Computation function to benchmark. If data_gen_fn is provided, this should
        accept the generated data as an argument. Otherwise, it takes no arguments.
    device : str, optional
        "cpu" or "cuda". Used to decide memory tracking.
    min_run_time : float, optional
        Minimum total run time in seconds.
    num_threads : int, optional
        Number of threads for the benchmark. Defaults to torch.get_num_threads().
    min_runs : int, optional
        Minimum number of runs to perform. Defaults to 10.
    data_gen_fn : callable, optional
        If provided, this function regenerates data each iteration (not timed).
        The computation fn should accept the generated data as an argument.

    Returns
    -------
    measurement : torch.utils.benchmark.Measurement
        Timing measurement.
    peak_mem_bytes : int or None
        Peak GPU memory allocated during the benchmark, or None for cpu.
    """
    if min_run_time is None:
        min_run_time = 0.5

    if num_threads is None:
        num_threads = torch.get_num_threads()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # If no data_gen_fn, use standard Timer approach
    if data_gen_fn is None:
        timer = benchmark.Timer(
            stmt="fn()",
            globals={"fn": fn},
            num_threads=num_threads,
        )
        measurement = timer.blocked_autorange(min_run_time=min_run_time)

        # Ensure minimum runs
        if len(measurement.times) < min_runs:
            for _ in range(min_runs - len(measurement.times)):
                elapsed = _benchmark_one(fn, device)
                measurement.raw_times.append(elapsed)
    else:
        # Manual timing with separate data generation
        times = []
        total_time = 0.0

        # Warmup
        data = data_gen_fn()
        fn(data)

        while total_time < min_run_time or len(times) < min_runs:
            # Generate data (not timed)
            data = data_gen_fn()
            elapsed = _benchmark_one(fn, device, data)
            times.append(elapsed)
            total_time += elapsed

        # Create a simple measurement object
        measurement = _SimpleMeasurement(times)

    peak_mem = None
    if device == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()

    return measurement, peak_mem


def _benchmark_one(fn, device, *args, **kwargs) -> float:
    """Run a single benchmark with the appropriate device tracking and return time in seconds."""
    if device == "cuda":
        # Use CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn(*args, **kwargs)
        end_event.record()
        end_event.synchronize()
        elapsed_s = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to s
    else:
        start = time.perf_counter()
        fn(*args, **kwargs)
        end = time.perf_counter()
        elapsed_s = end - start
    return elapsed_s


class _SimpleMeasurement:
    """Simple measurement object for manual timing."""

    def __init__(self, times):
        self.raw_times = times
        self.number_per_run = 1
        self._times = times

    @property
    def times(self):
        return self._times

    @property
    def mean(self):
        return sum(self._times) / len(self._times) if self._times else 0.0

    @property
    def median(self):
        if not self._times:
            return 0.0
        sorted_times = sorted(self._times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return sorted_times[n // 2]

    @property
    def iqr(self):
        if len(self._times) < 4:
            return 0.0
        sorted_times = sorted(self._times)
        n = len(sorted_times)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_times[q1_idx]
        q3 = sorted_times[q3_idx]
        return q3 - q1
