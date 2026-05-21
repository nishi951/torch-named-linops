"""CUDA execution tracing for torchlinops.

Provides a lightweight logger that records CUDA synchronization events
(record_event, wait_event, wait_stream) as a labeled dependency graph.
Independent of actual CUDA Event/Stream objects — callers provide labels.

Usage::

    import torchlinops.config as config
    from torchlinops.cuda_trace import cuda_logger

    config.log_cuda_events = True

    # ... run linops ...

    print(cuda_logger.display())
    cuda_logger.reset()
"""

from collections import deque
from dataclasses import dataclass, field

import torch

__all__ = ["CUDALogger", "cuda_logger"]


@dataclass
class CUDALogger:
    """Tracks logical CUDA synchronization as a labeled DAG.

    Independent of actual CUDA Event/Stream objects. Callers provide
    string labels and torch.device values; the logger builds a
    dependency graph that can be displayed as an ASCII timeline.

    Symbols in the output:
        ● record  = event recorded (marks completion of an operation)
        ◇ wait    = stream waiting for another operation to complete
        ○ implicit = auto-created node (e.g., default stream at start of operation)
        ←         = "blocks on" — this node cannot start until the target completes

    The timeline groups nodes into stages using topological sort.
    Operations in the same stage can execute in parallel.
    Operations in later stages are blocked on earlier stages.
    """

    _nodes: list[dict] = field(default_factory=list)
    _edges: list[tuple] = field(default_factory=list)
    _next_id: int = 0
    _implicit: dict = field(default_factory=dict)

    def record(self, label: str, device: torch.device) -> int:
        """Log an event recording (marks operation completion).

        Parameters
        ----------
        label : str
            Human-readable label for this event.
        device : torch.device
            Device where the event was recorded.

        Returns
        -------
        int
            Node ID for use in wait() calls.
        """
        nid = self._next_id
        self._next_id += 1
        self._nodes.append(
            {"id": nid, "label": label, "device": device, "type": "record"}
        )
        return nid

    def wait(
        self, label: str, device: torch.device, targets: list[int], reason: str = ""
    ) -> int:
        """Log a wait operation (stream waiting for other events).

        Parameters
        ----------
        label : str
            Human-readable label for this wait.
        device : torch.device
            Device where the wait occurs.
        targets : list[int]
            Node IDs this wait depends on.
        reason : str
            Why we're waiting (e.g., "end_event", "wait_stream").

        Returns
        -------
        int
            Node ID for this wait node.
        """
        nid = self._next_id
        self._next_id += 1
        self._nodes.append(
            {"id": nid, "label": label, "device": device, "type": "wait"}
        )
        for tid in targets:
            self._edges.append((nid, tid, reason))
        return nid

    def implicit_node(self, label: str, device: torch.device) -> int:
        """Create or reuse an implicit node (e.g., default stream).

        Implicit nodes represent things that aren't explicitly recorded
        but are dependency sources (e.g., "the default stream at the start
        of this operation"). They are deduplicated by (label, device).

        Parameters
        ----------
        label : str
            Label for the implicit node.
        device : torch.device
            Device for the implicit node.

        Returns
        -------
        int
            Node ID.
        """
        key = (label, str(device))
        if key not in self._implicit:
            nid = self._next_id
            self._next_id += 1
            self._nodes.append(
                {"id": nid, "label": label, "device": device, "type": "implicit"}
            )
            self._implicit[key] = nid
        return self._implicit[key]

    def display(self, reset: bool = False) -> str:
        """Return an ASCII representation of the logged CUDA events.

        Shows:
        1. Legend explaining symbols
        2. Nodes grouped by device
        3. Execution timeline (topologically sorted stages)
        4. Summary of parallelism
        """
        if not self._nodes:
            return "CUDA Execution Trace\n====================\n(no events logged)"

        lines = []
        lines.append("CUDA Execution Trace")
        lines.append("====================")
        lines.append("")
        lines.append("Legend:")
        lines.append("  ● record   = event recorded (marks completion of an operation)")
        lines.append("  ◇ wait     = stream waiting for another operation to complete")
        lines.append(
            "  ○ implicit = auto-created node (e.g., default stream at start of operation)"
        )
        lines.append(
            "  ←          = 'blocks on' — this node cannot start until the target completes"
        )
        lines.append("")

        # Group nodes by device
        devices = {}
        for n in self._nodes:
            d = str(n["device"])
            devices.setdefault(d, []).append(n)

        lines.append("Nodes (grouped by device):")
        for dev in sorted(devices.keys()):
            lines.append(f"  {dev}:")
            for n in devices[dev]:
                symbol = (
                    "●"
                    if n["type"] == "record"
                    else "○"
                    if n["type"] == "implicit"
                    else "◇"
                )
                deps = [f"[{t}]" for w, t, _ in self._edges if w == n["id"]]
                dep_str = f" ← {', '.join(deps)}" if deps else ""
                lines.append(f"    [{n['id']}] {symbol} {n['label']}{dep_str}")
            lines.append("")

        # Build adjacency for topological sort
        node_ids = {n["id"] for n in self._nodes}
        in_degree = {nid: 0 for nid in node_ids}
        successors = {nid: [] for nid in node_ids}
        for waiter, target, _ in self._edges:
            if waiter in node_ids and target in node_ids:
                in_degree[waiter] += 1
                successors[target].append(waiter)

        queue = deque(nid for nid in node_ids if in_degree[nid] == 0)
        stages = []
        while queue:
            group = sorted(queue)
            stages.append(group)
            next_queue = deque()
            for nid in group:
                for succ in successors[nid]:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        next_queue.append(succ)
            queue = next_queue

        node_map = {n["id"]: n for n in self._nodes}

        lines.append("Execution Timeline:")
        for s_idx, stage in enumerate(stages):
            blocked = s_idx > 0
            blocker_label = (
                f" (blocked on Stage {s_idx - 1})"
                if blocked
                else " (parallel — no blockers)"
            )
            lines.append(f"  ═══ Stage {s_idx} ═══{blocker_label}")
            for nid in stage:
                n = node_map[nid]
                deps = [(t, r) for w, t, r in self._edges if w == nid]
                dep_str = ""
                if deps:
                    dep_parts = []
                    for t, _ in deps:
                        dep_parts.append(f"[{t}]")
                    dep_str = f" ← blocks on {', '.join(dep_parts)}"
                lines.append(f"    [{nid}] {n['device']}  {n['label']}{dep_str}")
            lines.append("")

        max_parallel = max(len(s) for s in stages) if stages else 0
        lines.append(
            f"⚠ {len(stages)} serial stages — max parallelism: {max_parallel} ops"
        )
        if reset:
            self.reset()

        return "\n".join(lines)

    def reset(self):
        """Clear all logged events."""
        self._nodes.clear()
        self._edges.clear()
        self._next_id = 0
        self._implicit.clear()


cuda_logger = CUDALogger()
