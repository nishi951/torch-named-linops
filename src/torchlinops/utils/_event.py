from dataclasses import dataclass, field
from typing import Any, Callable, List
from collections import defaultdict

__all__ = [
    "EventManager",
    "Callback",
]


@dataclass
class Callback:
    name: str
    fn: Callable
    deps: List[str] = field(default_factory=lambda: [])


class EventManager:
    def __init__(self, handlers: dict = None):
        """
        handlers : defaultdict(list)
            defaultdict mapping event names to lists of (callback_name, callback_deps, callback_fn) triples
        """
        self.handlers = defaultdict(list) if handlers is None else handlers
        for event, callbacks in self.handlers.items():
            self.handlers[event] = self.sort_callbacks(callbacks)

    def register_handler(
        self,
        event,
        callback: Callback,
        sort_now: bool = True,
    ):
        self.handlers[event].append(callback)
        if sort_now:
            self.handlers[event] = self.sort_callbacks(self.handlers[event])

    def dispatch(self, event, s: Any):
        for handler in self.handlers[event]:
            s = handler.fn(s)
        return s

    def handlers(self, event):
        return [handler.name for handler in self.handlers[event]]

    @staticmethod
    def sort_callbacks(callbacks):
        """ChatGPT"""
        # Create a mapping from callback name to callback instance
        callback_map = {cb.name: cb for cb in callbacks}

        # Track visited callbacks to avoid cycles and repeated visits
        visited = set()
        # Use a list to act as an ordered stack for the sorted elements
        sorted_callbacks = []

        def dfs(cb: Callback):
            if cb.name in visited:
                return
            visited.add(cb.name)
            # Visit all dependencies first
            for dep_name in cb.deps:
                if dep_name in callback_map:
                    dfs(callback_map[dep_name])
                else:
                    raise ValueError(
                        f"Dependency {dep_name} not found for callback {cb.name}"
                    )
            # Add this callback to the sorted list
            sorted_callbacks.append(cb)

        # Iterate through all callbacks and perform DFS
        for cb in callbacks:
            if cb.name not in visited:
                dfs(cb)

        return sorted_callbacks
