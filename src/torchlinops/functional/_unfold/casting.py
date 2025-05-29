try:
    import triton
    import triton.language as tl

    TRITON_ENABLED = True
except ImportError:
    from torchlinops.utils import fake_triton as triton, fake_tl as tl

    TRITON_ENABLED = False

__all__ = ["scalar_cast"]


@triton.jit
def scalar_cast(t, dtype):
    """Replaces .to(). Necessary to avoid certain optimizations
    in cases when the inputs to the function are 1
    """
    return tl.full([], t, dtype)
