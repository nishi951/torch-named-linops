from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class Forwarded:
    allow_set_upstream: bool = False
    """If true, allow setting this value to affect upstream values."""
    _value: Optional[Any] = None
    _obj: Optional[Any] = None
    _attr: Optional[Any] = None

    @property
    def value(self):
        if self._obj is None:
            # No object to forward to
            return self._value
        elif self._value is not None:
            # A preset value overrides forwarded reference
            return self._value
        return getattr(self._obj, self._attr)

    @value.setter
    def value(self, new_value):
        if self._obj is None:
            # No object to forward to
            self._value = new_value
        elif self.allow_set_upstream:
            setattr(self._obj, self._attr)
        else:
            # Don't overwrite upstream
            self._value = new_value

            # Clear pointer
            self._obj = None
            self._attr = None

    def forward_to(self, obj, attr):
        """Create reference"""
        self._obj = obj
        self._attr = attr
        self._value = None  # Reset value


class MyClass:
    def __init__(self):
        self._start_event = Forwarded()
        self._end_event = Forwarded()

    @property
    def start_event(self):
        return self._start_event.value

    @start_event.setter
    def start_event(self, event):
        """
        Parameters
        ----------
        event : Event | tuple[Any, str]
            If a bare Event is provided, use that event on this object.
            If a tuple, interpret it as a reference to forward. e.g. event = (other_linop, 'start_event')
            will forward this linop's linop.start_event to other_linop.start_event
        """
        if isinstance(event, tuple):
            self._start_event.forward_to(*event)
        else:
            self._start_event.value = event

    @property
    def end_event(self):
        return self._end_event.value

    @end_event.setter
    def end_event(self, event):
        if isinstance(event, tuple):
            self._end_event.forward_to(*event)
        else:
            self._end_event.value = event


def main():
    inst1 = MyClass()
    inst2 = MyClass()
    inst3 = MyClass()

    print("test 1 - basic")
    inst1.start_event = "foo"  # Bare value
    inst1.end_event = (inst2, "start_event")  # inst1.end_event -> inst2.start_event
    print(inst1.start_event)  # foo
    print(inst1.end_event)  # None

    print("test 2 - changing forwarded values")
    inst2.start_event = "bar"
    print(inst1.end_event)  # bar

    print("test 3 - multi-hop forwarding")
    # inst1.end_event -> inst2.start_event -> inst3.start_event
    inst2.start_event = (inst3, "end_event")
    print(inst1.end_event)  # None
    inst3.end_event = "baz"
    print(inst1.end_event)  # baz


if __name__ == "__main__":
    main()
