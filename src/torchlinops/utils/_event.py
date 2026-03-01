import torch

__all__ = ["RepeatedEvent", "assert_gpus_overlap"]


class RepeatedEvent:
    """Manage a FIFO queue of CUDA events for stream synchronization.

    Keeps only the most recent event, dropping old references to free
    resources. The wrapper itself can be passed directly to wait_event().
    """

    def __init__(self, **event_kwargs):
        """
        A wrapper so each record() creates a fresh CUDA event,
        but the wrapper itself can be passed directly to wait_event().

        Parameters
        ----------
        **event_kwargs
            Keyword arguments passed to ``torch.cuda.Event(...)``.
        """
        self._event_kwargs = event_kwargs
        self._last_event = None

    def record(self, stream=None):
        """
        Create a new CUDA event and record it on the given stream.
        Old events are dropped immediately to free resources.
        """
        # Drop old event reference
        self._last_event = None

        # Create and record new event
        ev = torch.cuda.Event(**self._event_kwargs)
        if stream is None:
            stream = torch.cuda.current_stream()
        ev.record(stream)

        # Store and return self for chaining
        self._last_event = ev
        return self

    @property
    def last_event(self):
        return self._last_event

    def __repr__(self):
        return f"<RepeatedEvent wrapping {self._last_event!r}>"


def _get_cuda_kernel_events(prof):
    """
    Extract CUDA kernel events from a torch.profiler profile object.
    """
    return [
        evt
        for evt in prof.events()
        if "CUDA" in str(evt.device_type) and evt.device_time_total > 0
    ]


def _merge_intervals(intervals):
    """
    Merge overlapping intervals.
    intervals: list of (start_ns, end_ns)
    Returns merged list.
    """
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _compute_overlap(intervals_a, intervals_b):
    """
    Compute total overlap (ns) between two interval lists.
    Assumes intervals are already merged.
    """
    i = j = 0
    total = 0

    while i < len(intervals_a) and j < len(intervals_b):
        a_start, a_end = intervals_a[i]
        b_start, b_end = intervals_b[j]

        start = max(a_start, b_start)
        end = min(a_end, b_end)

        if start < end:
            total += end - start

        # Advance the interval that ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1

    return total


def assert_gpus_overlap(
    prof,
    *,
    min_overlap_ms: float = 5.0,
    min_overlap_ratio: float | None = None,
):
    """
    Assert that at least two GPUs executed concurrently.

    Args:
        prof: torch.profiler.profile object
        min_overlap_ms: minimum absolute overlap required (default 5ms)
        min_overlap_ratio: optional minimum fraction of smaller GPU runtime
                           that must overlap (e.g. 0.2 for 20%)
    """

    kernels = _get_cuda_kernel_events(prof)

    if not kernels:
        raise AssertionError("No CUDA kernel events found in profiler trace")

    # Group intervals by device
    by_device = {}
    for evt in kernels:
        start = evt.time_range.start
        end = evt.time_range.end
        by_device.setdefault(evt.device_index, []).append((start, end))

    if len(by_device) < 2:
        raise AssertionError("Need kernels from at least two GPUs")

    # Merge intervals per device
    merged = {dev: _merge_intervals(intervals) for dev, intervals in by_device.items()}

    devices = sorted(merged.keys())

    # Check all device pairs
    found_valid_overlap = False
    diagnostic = []

    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            d0, d1 = devices[i], devices[j]

            overlap_ns = _compute_overlap(
                merged[d0],
                merged[d1],
            )

            total0 = sum(e - s for s, e in merged[d0])
            total1 = sum(e - s for s, e in merged[d1])
            smaller_total = min(total0, total1)

            overlap_ms = overlap_ns / 1e6
            ratio = overlap_ns / smaller_total if smaller_total > 0 else 0.0

            diagnostic.append(
                f"GPU {d0} vs {d1}: "
                f"{overlap_ms:.3f} ms overlap "
                f"({ratio:.2%} of smaller runtime)"
            )

            abs_ok = overlap_ms >= min_overlap_ms
            ratio_ok = True if min_overlap_ratio is None else ratio >= min_overlap_ratio

            if abs_ok and ratio_ok:
                found_valid_overlap = True

    if not found_valid_overlap:
        diag_str = "\n".join(diagnostic)
        raise AssertionError(
            "No sufficient GPU concurrency detected.\n"
            f"Criteria: min_overlap_ms={min_overlap_ms}, "
            f"min_overlap_ratio={min_overlap_ratio}\n"
            f"Diagnostics:\n{diag_str}"
        )
