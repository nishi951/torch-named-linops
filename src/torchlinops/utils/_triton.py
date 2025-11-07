from functools import wraps
from types import SimpleNamespace
from typing import Any

__all__ = ["fake_triton", "fake_tl"]


# Replace all triton decorators with not implemented
def not_implemented_wrapper(*args, **kwargs):
    raise NotImplementedError()


def not_implemented_decorator(f):
    return wraps(f)(not_implemented_wrapper)


fake_triton = SimpleNamespace()
fake_triton.jit = lambda f: wraps(f)(not_implemented_wrapper)
fake_triton.heuristics = lambda *args, **kwargs: not_implemented_decorator

fake_tl = SimpleNamespace()
fake_tl.constexpr = Any
fake_tl.float32 = float
