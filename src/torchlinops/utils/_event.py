import torch

__all__ = ["RepeatedEvent"]


class RepeatedEvent:
    def __init__(self, **event_kwargs):
        """
        A wrapper so each record() creates a fresh CUDA event,
        but the wrapper itself can be passed directly to wait_event().

        Args:
            event_kwargs: Passed to torch.cuda.Event(...)
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
