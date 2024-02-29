from dataclasses import dataclass, field


## Some useful handlers
@dataclass
class AbstractTrainingCallback:
    dependencies: List = field(default_factory=lambda: [])

    def __call__(self, s: TrainingState):
        raise NotImplementedError()


class GlobalStep(AbstractTrainingCallback):
    def __call__(self, s: TrainingState):
        s.global_step += 1
        return s


class EpochStep(AbstractTrainingCallback):
    def __call__(self, s: TrainingState):
        s.epoch += 1
        return s

def topological_sort_callbacks(callbacks):
    """ChatGPT"""
    # Create a mapping from callback name to callback instance
    callback_map = {cb.__class__.__name__: cb for cb in callbacks}

    # Track visited callbacks to avoid cycles and repeated visits
    visited = set()
    # Use a list to act as an ordered stack for the sorted elements
    sorted_callbacks = []

    def dfs(callback):
        if callback.__class__.__name__ in visited:
            return
        visited.add(callback.name)
        # Visit all dependencies first
        for dep_name in callback.dependencies:
            if dep_name in callback_map:
                dfs(callback_map[dep_name])
            else:
                raise ValueError(
                    f'Dependency {dep_name} not found for callback {callback.name}'
                )
        # Add this callback to the sorted list
        sorted_callbacks.append(callback)

    # Iterate through all callbacks and perform DFS
    for cb in callbacks:
        if cb.name not in visited:
            dfs(cb)

    # Return the reversed list since the last added should be the first executed
    return list(reversed(sorted_callbacks))
