try:
    import triton
    import triton.language as tl

    TRITON_ENABLED = True
except ImportError:
    from types import SimpleNamespace
    from functools import wraps

    # Replace all triton decorators with not implemented
    def not_implemented_wrapper(*args, **kwargs):
        raise NotImplementedError()

    triton = SimpleNamespace()
    triton.jit = lambda f: wraps(f)(not_implemented_wrapper)
    triton.heuristics = lambda f: wraps(f)(not_implemented_wrapper)

    TRITON_ENABLED = False

__all__ = ["scalar_cast"]


@triton.jit
def scalar_cast(t, dtype):
    """Replaces .to(). Necessary to avoid certain optimizations
    in cases when the inputs to the function are 1
    """
    return tl.full([], t, dtype)
