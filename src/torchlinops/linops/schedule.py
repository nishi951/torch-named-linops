"""Execution schedule for composite linops.

Provides a declarative dependency graph that each composite linop
(Add, Concat, Stack, Chain) uses to coordinate parallel execution
of its direct children.

Schedules are hierarchical — nested composites have their own schedules.
Each level manages its own synchronization independently.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.cuda import Event, Stream, default_stream

from torchlinops.utils import default_to

__all__ = ["ExecutionSchedule", "execute_schedule", "schedule_to_ascii_dag"]


@dataclass(frozen=True)
class ExecutionSchedule:
    """Immutable execution plan for a composite linop's direct children.

    Each composite linop owns one schedule describing the dependency
    relationships among its immediate children. Schedules are hierarchical —
    nested composites have their own schedules.

    Parameters
    ----------
    dependencies : dict[int, list[tuple[int, str]]]
        For each child index: list of (dependency_index, event_name) that
        must complete before this child can start. Empty list means no
        dependencies — the child starts immediately.

    Examples
    --------
    Sequential (Chain): child 1 waits for child 0, child 2 waits for child 1::

        ExecutionSchedule({
            0: [],
            1: [(0, "end_event")],
            2: [(1, "end_event")],
        })

    Parallel (Add): all children start immediately::

        ExecutionSchedule({
            0: [],
            1: [],
            2: [],
        })
    """

    dependencies: dict[int, list[tuple[int, str]]]

    @property
    def is_sequential(self) -> bool:
        """True if every child (except the first) depends on the previous one."""
        deps = self.dependencies
        indices = sorted(deps.keys())
        if len(indices) <= 1:
            return True
        for i, idx in enumerate(indices[1:], 1):
            expected_dep = (indices[i - 1], "end_event")
            if deps[idx] != [expected_dep]:
                return False
        return deps.get(indices[0], []) == []

    @property
    def is_parallel(self) -> bool:
        """True if all children have no dependencies."""
        return all(deps == [] for deps in self.dependencies.values())

    def to_dict(self) -> dict:
        """Serializable dict representation."""
        return {str(k): [(d, e) for d, e in v] for k, v in self.dependencies.items()}

    def __repr__(self) -> str:
        deps_str = ", ".join(f"{k}: {v}" for k, v in sorted(self.dependencies.items()))
        return f"ExecutionSchedule({{{deps_str}}})"


def topological_groups(
    dependencies: dict[int, list[tuple[int, str]]],
) -> list[list[int]]:
    """Return execution order as list of groups.

    Each group contains indices that can run in parallel.
    Groups are ordered so that all dependencies of group N
    are satisfied by groups 0..N-1.

    Uses Kahn's algorithm, grouping all ready nodes together.

    Parameters
    ----------
    dependencies : dict[int, list[tuple[int, str]]]
        Dependency graph: index -> list of (dep_index, event_name).

    Returns
    -------
    list[list[int]]
        Execution order as groups of parallel indices.

    Examples
    --------
    >>> topological_groups({0: [], 1: [], 2: [(0, "end_event")]})
    [[0, 1], [2]]
    """
    if not dependencies:
        return []

    # Build in-degree count (only count deps within this schedule)
    all_indices = set(dependencies.keys())
    in_degree = {idx: 0 for idx in all_indices}
    successors = {idx: [] for idx in all_indices}

    for idx, deps in dependencies.items():
        for dep_idx, _event_name in deps:
            if dep_idx in all_indices:
                in_degree[idx] += 1
                successors[dep_idx].append(idx)

    # Start with nodes that have no dependencies
    queue = deque(idx for idx in all_indices if in_degree[idx] == 0)
    groups = []

    while queue:
        # All nodes currently in queue can run in parallel
        group = sorted(queue)
        groups.append(group)

        next_queue = deque()
        for idx in group:
            for succ in successors[idx]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    next_queue.append(succ)
        queue = next_queue

    if sum(len(g) for g in groups) != len(all_indices):
        raise ValueError(f"Cyclic dependency detected in schedule: {dependencies}")

    return groups


def execute_schedule(
    parent,
    schedule: ExecutionSchedule,
    x: Tensor,
    reduce_fn: Callable[[list[Tensor]], Tensor],
    threaded: bool = True,
    num_workers: Optional[int] = None,
    linops_override: Optional[list] = None,
) -> Tensor:
    """Execute children according to the schedule.

    Parameters
    ----------
    parent : NamedLinop
        The composite linop whose children to execute.
    schedule : ExecutionSchedule
        The dependency graph for this level.
    x : Tensor
        Input tensor.
    reduce_fn : Callable[[list[Tensor]], Tensor]
        Function to combine child outputs (sum, cat, stack, etc.).
    threaded : bool
        Whether to use ThreadPoolExecutor for parallel groups.
    num_workers : int | None
        Max worker threads. If None, defaults to group size.
    linops_override : list | None
        Alternative list of linops to execute instead of parent._linops.
        Used for adj_fn where adjoint linops differ from forward linops.

    Returns
    -------
    Tensor
        Combined result from all children.
    """
    deps = schedule.dependencies
    indices = sorted(deps.keys())
    linops = linops_override if linops_override is not None else parent._linops

    if not indices:
        raise ValueError("Cannot execute schedule with no children")

    # CPU path: simple sequential execution
    if not x.is_cuda:
        results = []
        for idx in indices:
            results.append(linops[idx](x))
        return reduce_fn(results)

    # CUDA path: event-synchronized execution
    stream: Stream = default_to(default_stream(x.device), parent.stream)

    # Local event tracking — fresh per forward call, never stored on linops
    events: dict[int, dict[str, Event]] = {}

    # Record parent start event
    parent.start_event = stream.record_event()

    # Compute execution groups
    groups = topological_groups(deps)

    # If all children are in one parallel group and threading is enabled
    if len(groups) == 1 and threaded:
        results = _execute_parallel_group(
            linops, groups[0], deps, events, x, stream, num_workers
        )
    else:
        # Sequential or mixed: execute groups in order
        results = []
        for group in groups:
            if len(group) == 1 and not threaded:
                # Single child, no threading
                idx = group[0]
                _wait_dependencies(idx, deps, events, stream)
                results.append(linops[idx](x))
            else:
                # Parallel group
                group_results = _execute_parallel_group(
                    linops, group, deps, events, x, stream, num_workers
                )
                results.extend(group_results)

    y = reduce_fn(results)

    # Record parent end event
    x.record_stream(stream)
    parent.end_event = stream.record_event()

    return y


def _wait_dependencies(
    idx: int,
    deps: dict[int, list[tuple[int, str]]],
    events: dict[int, dict[str, Event]],
    stream: Stream,
) -> None:
    """Wait for all declared dependencies of a child to complete."""
    for dep_idx, event_name in deps.get(idx, []):
        dep_event = events[dep_idx][event_name]
        stream.wait_event(dep_event)


def _execute_parallel_group(
    linops,
    group: list[int],
    deps: dict[int, list[tuple[int, str]]],
    events: dict[int, dict[str, Event]],
    x: Tensor,
    stream: Stream,
    num_workers: Optional[int],
) -> list[Tensor]:
    """Execute a group of children, respecting intra-group dependencies."""
    if num_workers is None:
        num_workers = len(group)

    results: list[Optional[Tensor]] = [None] * len(group)

    def worker(pos_idx: int):
        idx = group[pos_idx]
        linop = linops[idx]

        # Wait for dependencies (from previous groups)
        _wait_dependencies(idx, deps, events, stream)

        # Execute child
        y = linop(x)

        # Record end event for this child
        end_event = stream.record_event()
        events[idx] = {"end_event": end_event}

        results[pos_idx] = y

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(worker, range(len(group))))

    return [r for r in results if r is not None]


def schedule_to_ascii_dag(
    schedule: ExecutionSchedule,
    child_names: Optional[list[str]] = None,
) -> str:
    """Render the schedule as an ASCII DAG diagram.

    Parameters
    ----------
    schedule : ExecutionSchedule
        The schedule to render.
    child_names : list[str], optional
        Names for each child index. If None, uses index numbers.

    Returns
    -------
    str
        ASCII representation of the dependency graph.
    """
    deps = schedule.dependencies
    indices = sorted(deps.keys())
    groups = topological_groups(deps)

    if child_names is None:
        child_names = [str(i) for i in indices]

    lines = []
    lines.append("Execution DAG:")

    for g_idx, group in enumerate(groups):
        prefix = f"  Group {g_idx} (parallel):"
        lines.append(prefix)
        for idx in group:
            name = child_names[idx] if idx < len(child_names) else str(idx)
            dep_list = deps.get(idx, [])
            if dep_list:
                dep_strs = [
                    f"{child_names[d] if d < len(child_names) else d}[{e}]"
                    for d, e in dep_list
                ]
                lines.append(f"    {name} <- {', '.join(dep_strs)}")
            else:
                lines.append(f"    {name} (no deps)")

    return "\n".join(lines)
