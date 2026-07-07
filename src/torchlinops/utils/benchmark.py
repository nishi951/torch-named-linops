# pragma: exclude file
import copy
import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

from ._log import Indenter

logger = logging.getLogger("torchmri.utils")

__all__ = ["benchmark", "benchmark_and_summarize", "BenchmarkResult"]


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.

    Attributes
    ----------
    times : list of float
        Per-run times in seconds.
    number_per_run : int
        Number of function executions per measurement block.
    peak_mem_bytes : int or None
        Peak GPU memory used during the benchmark, in bytes.
    """

    times: list[float] = field(default_factory=list)
    number_per_run: int = 1
    peak_mem_bytes: Optional[int] = None

    @property
    def mean(self) -> float:
        """Mean time per run in seconds."""
        return float(np.mean(self.times)) if self.times else 0.0

    @property
    def median(self) -> float:
        """Median time per run in seconds."""
        return float(np.median(self.times)) if self.times else 0.0

    @property
    def iqr(self) -> float:
        """Interquartile range of per-run times in seconds."""
        if len(self.times) < 4:
            return 0.0
        q75, q25 = np.percentile(self.times, [75, 25])
        return float(q75 - q25)


def benchmark_and_summarize(
    fn,
    *args,
    num_iters: int = 10,
    ignore_first: int = 0,
    backend: Literal["torch", "cupy"] = "torch",
    name: str = None,
    **kwargs,
):
    """Convenience function"""
    result, output = benchmark(
        fn, *args, num_iters=num_iters, backend=backend, **kwargs
    )
    if name is None:
        name = fn.__name__
    summarize(result, name=name, ignore_first=ignore_first)
    return result, output


def benchmark(
    fn,
    *args,
    num_iters: int = 10,
    backend: Literal["torch", "cupy"] = "torch",
    **kwargs,
):
    """Benchmark a function called with some arguments.

    Defaults to torch benchmarking
    """
    if backend == "torch":
        handler = TorchHandler()
    elif backend == "cupy":
        handler = CupyHandler()
    else:
        raise ValueError(f"Unrecognized backend type {backend}")
    fn_result = fn(*args, **kwargs)  # Warmup
    handler.bench_start()
    for _ in range(num_iters):
        handler.trial_start()
        fn(*args, **kwargs)
        handler.trial_end()
    handler.bench_end()
    return handler.result, fn_result


def summarize(benchmark_result, name: str, ignore_first: int = 0):
    """Summarize the results from benchmark"""
    with Indenter() as indent:
        print(name)
        with indent:
            timings_ms = benchmark_result["timings_ms"][ignore_first:]
            indent.print(f"Mean Time: {np.mean(timings_ms):0.3f} ms")
            indent.print(f"Min Time: {np.min(timings_ms):0.3f} ms")
            indent.print(f"Max Time: {np.max(timings_ms):0.3f} ms")
            indent.print(f"Memory: {benchmark_result['max_mem_bytes']} bytes")


class TorchHandler:
    """Benchmark handler for PyTorch operations.

    Supports both CPU and CUDA timing with memory tracking.
    """

    def __init__(
        self,
        device: str = "cuda",
        memory_snapshot_file: Optional[Path] = None,
    ):
        self.device = device
        self.memory_snapshot_file = memory_snapshot_file
        self.reset()

    def reset(self):
        self._start_event = None
        self._end_event = None
        self._start_time = None

        self.result = {
            "timings_ms": [],
            "max_mem_bytes": None,
        }

    def bench_start(self, *args, **kwargs):
        self.reset()
        gc.disable()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.memory._record_memory_history(max_entries=100000)

    def bench_end(self, *args, **kwargs):
        if self.device == "cuda":
            self.result["max_mem_bytes"] = torch.cuda.max_memory_allocated()
        gc.enable()
        logger.info(f"Max memory allocated: {self.result['max_mem_bytes']}")
        if self.memory_snapshot_file is not None and self.device == "cuda":
            try:
                torch.cuda.memory._dump_snapshot(f"{str(self.memory_snapshot_file)}")
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot {e}")
            torch.cuda.memory._record_memory_history(enabled=None)

    def trial_start(self, event=None, i=None):
        if self.device == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()

    def trial_end(self, event=None, i=None):
        if self.device == "cuda":
            self._end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        logger.debug(f"{event}: {elapsed_ms}")
        self.result["timings_ms"].append(elapsed_ms)

    def blocked_autorange(
        self,
        fn,
        min_run_time: float = 0.5,
        min_runs: int = 10,
        data_gen_fn=None,
        num_warmup: int = 1,
    ) -> BenchmarkResult:
        """Run benchmark with automatic iteration count determination.

        Parameters
        ----------
        fn : callable
            Function to benchmark. If data_gen_fn is provided, fn should accept
            the generated data as an argument.
        min_run_time : float
            Minimum total run time in seconds.
        min_runs : int
            Minimum number of runs to perform.
        data_gen_fn : callable, optional
            If provided, regenerates data each iteration (not timed).
            The fn should accept the generated data as an argument.
        num_warmup : int
            Number of untimed warmup iterations before measurement.

        Returns
        -------
        BenchmarkResult
            Result containing timings and memory statistics.
        """
        for _ in range(num_warmup):
            if data_gen_fn is not None:
                data = data_gen_fn()
                fn(data)
            else:
                fn()

        if self.device == "cuda":
            torch.cuda.synchronize()

        self.bench_start()

        if data_gen_fn is not None:
            # Per-iteration timing with separate data generation
            total_time = 0.0
            while (
                total_time < min_run_time or len(self.result["timings_ms"]) < min_runs
            ):
                data = data_gen_fn()
                self.trial_start()
                fn(data)
                self.trial_end()
                total_time += self.result["timings_ms"][-1] / 1000.0
        else:
            # Blocked autorange: auto-calibrate runs_per_block
            runs_per_block = 1
            while True:
                self._start_event = (
                    torch.cuda.Event(enable_timing=True)
                    if self.device == "cuda"
                    else None
                )
                self._end_event = (
                    torch.cuda.Event(enable_timing=True)
                    if self.device == "cuda"
                    else None
                )

                if self.device == "cuda":
                    self._start_event.record()
                    for _ in range(runs_per_block):
                        fn()
                    self._end_event.record()
                    self._end_event.synchronize()
                    block_time_ms = self._start_event.elapsed_time(self._end_event)
                else:
                    start = time.perf_counter()
                    for _ in range(runs_per_block):
                        fn()
                    block_time_ms = (time.perf_counter() - start) * 1000.0

                if block_time_ms > 5.0:
                    break
                runs_per_block *= 2
                if runs_per_block > 1000000:
                    break

            # Collect measurements
            total_time = 0.0
            while (
                total_time < min_run_time or len(self.result["timings_ms"]) < min_runs
            ):
                if self.device == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    for _ in range(runs_per_block):
                        fn()
                    end_event.record()
                    end_event.synchronize()
                    block_time_ms = start_event.elapsed_time(end_event)
                else:
                    start = time.perf_counter()
                    for _ in range(runs_per_block):
                        fn()
                    block_time_ms = (time.perf_counter() - start) * 1000.0

                per_run_ms = block_time_ms / runs_per_block
                self.result["timings_ms"].append(per_run_ms)
                total_time += block_time_ms / 1000.0

        self.bench_end()

        # Convert to seconds and build result
        times_s = [t / 1000.0 for t in self.result["timings_ms"]]
        return BenchmarkResult(
            times=times_s,
            number_per_run=1,
            peak_mem_bytes=self.result["max_mem_bytes"],
        )

    def collect_results(self, event, data):
        return {"torch": copy.deepcopy(self.result)}


class CupyHandler:
    """Benchmark handler for CuPy-based operations.

    Uses cupy CUDA events for timing and cupy memory pool for memory tracking.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._start_event = None
        self._end_event = None
        self._mempool = None

        self.result = {
            "timings_ms": [],
            "max_mem_bytes": None,
        }

    def bench_start(self, *args, **kwargs):
        import cupy as cp

        self.reset()
        gc.disable()
        self._mempool = cp.get_default_memory_pool()
        self._mempool.free_all_blocks()

    def bench_end(self, *args, **kwargs):
        self.result["max_mem_bytes"] = self._mempool.total_bytes()
        gc.enable()

    def trial_start(self, event=None, i=None):
        import cupy as cp

        self._start_event = cp.cuda.Event(disable_timing=False)
        self._end_event = cp.cuda.Event(disable_timing=False)
        self._start_event.record()

    def trial_end(self, event=None, i=None):
        import cupy as cp

        self._end_event.record()
        self._end_event.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(self._start_event, self._end_event)
        self.result["timings_ms"].append(elapsed_ms)

    def blocked_autorange(
        self,
        fn,
        min_run_time: float = 0.5,
        min_runs: int = 10,
        data_gen_fn=None,
        num_warmup: int = 1,
    ) -> BenchmarkResult:
        """Run benchmark with automatic iteration count determination.

        Parameters
        ----------
        fn : callable
            Function to benchmark. If data_gen_fn is provided, fn should accept
            the generated data as an argument.
        min_run_time : float
            Minimum total run time in seconds.
        min_runs : int
            Minimum number of runs to perform.
        data_gen_fn : callable, optional
            If provided, regenerates data each iteration (not timed).
            The fn should accept the generated data as an argument.
        num_warmup : int
            Number of untimed warmup iterations before measurement.

        Returns
        -------
        BenchmarkResult
            Result containing timings and memory statistics.
        """
        import cupy as cp

        for _ in range(num_warmup):
            if data_gen_fn is not None:
                data = data_gen_fn()
                fn(data)
            else:
                fn()

        cp.cuda.runtime.deviceSynchronize()

        self.bench_start()

        if data_gen_fn is not None:
            # Per-iteration timing with separate data generation
            total_time = 0.0
            while (
                total_time < min_run_time or len(self.result["timings_ms"]) < min_runs
            ):
                data = data_gen_fn()
                self.trial_start()
                fn(data)
                self.trial_end()
                total_time += self.result["timings_ms"][-1] / 1000.0
        else:
            # Blocked autorange: auto-calibrate runs_per_block
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

            # Collect measurements
            total_time = 0.0
            while (
                total_time < min_run_time or len(self.result["timings_ms"]) < min_runs
            ):
                start = cp.cuda.Event(disable_timing=False)
                end = cp.cuda.Event(disable_timing=False)
                start.record()
                for _ in range(runs_per_block):
                    fn()
                end.record()
                end.synchronize()
                block_time_ms = cp.cuda.get_elapsed_time(start, end)

                per_run_ms = block_time_ms / runs_per_block
                self.result["timings_ms"].append(per_run_ms)
                total_time += block_time_ms / 1000.0

                # Track peak memory
                current_mem = self._mempool.total_bytes()
                if current_mem > (self.result["max_mem_bytes"] or 0):
                    self.result["max_mem_bytes"] = current_mem

        self.bench_end()

        # Convert to seconds and build result
        times_s = [t / 1000.0 for t in self.result["timings_ms"]]
        return BenchmarkResult(
            times=times_s,
            number_per_run=1,
            peak_mem_bytes=self.result["max_mem_bytes"],
        )

    def collect_results(self, *args, **kwargs):
        return {"cupy": copy.deepcopy(self.result)}
