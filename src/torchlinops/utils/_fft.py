import torch
import torch.fft as fft
from torch import Tensor
from torch.autograd import Function

__all__ = ["cfft", "cifft", "cfft2", "cifft2", "cfftn", "cifftn"]


def cfftn(x, dim=None, norm="ortho", method="shift"):
    """Compute the centered n-dimenional FFT.

    Assumes the origin lies in the middle of the array (i.e., that the array has
    been fftshifted)

    Parameters
    ----------
    dim : tuple[int, ...]
        The dimensions over which to take the ifft.
    norm : norm (str, optional)
        Normalization mode. For the forward transform (fft()), these correspond to:

        - "forward" - normalize by 1/n
        - "backward" - no normalization
        - "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)

        Calling the backward transform (cifftn()) with the same normalization
        mode will apply an overall normalization of 1/n between the two transforms.
        This is required to make ifft() the exact inverse. Default is "backward"
        (no normalization).
    method : str
    """
    return CenteredFFTFn.apply(x, dim, norm, method)


def cifftn(x, dim=None, norm="ortho", method="shift"):
    """Compute the centered n-dimensional inverse FFT.

    Assumes the origin lies in the middle of the array (i.e., that the array has
    been fftshifted)

    Parameters
    ----------
    dim : tuple[int, ...]
        The dimensions over which to take the ifft.
    norm : norm (str, optional)
        Normalization mode. For the backward transform (ifft()), these correspond to:

        - "forward" - no normalization
        - "backward" - normalize by 1/n
        - "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)

        Calling the forward transform (cfftn()) with the same normalization mode
        will apply an overall normalization of 1/n between the two transforms. This
        is required to make ifft() the exact inverse. Default is "backward"
        (normalize by 1/n).
    """
    return CenteredIFFTFn.apply(x, dim, norm, method)


class CenteredFFTFn(Function):
    """Memory-efficient centered fftn."""

    @staticmethod
    def forward(
        x: Tensor,
        dim: tuple[int, ...],
        norm: str,
        method: str,
    ) -> Tensor:
        return _cfftn(x, dim, norm, method)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.dim, ctx.norm, ctx.method = inputs

    @staticmethod
    def backward(ctx, grad_output):
        # Swap normalization mode to achieve adjoint behavior
        if ctx.norm == "forward":
            backward_norm = "backward"
        elif ctx.norm == "backward":
            backward_norm = "forward"
        elif ctx.norm == "ortho":
            backward_norm = "ortho"
        else:
            raise ValueError(f"Unknown fft normalization: {ctx.norm}")
        return (
            _cifftn(grad_output, ctx.dim, backward_norm, ctx.method),
            None,
            None,
            None,
        )


class CenteredIFFTFn(Function):
    """Equal to ungrid"""

    @staticmethod
    def forward(
        x: Tensor,
        dim: tuple[int, ...],
        norm: str,
        method: str,
    ) -> Tensor:
        return _cifftn(x, dim, norm, method)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.dim, ctx.norm, ctx.method = inputs

    @staticmethod
    def backward(ctx, grad_output):
        # Swap normalization mode to achieve adjoint behavior
        if ctx.norm == "forward":
            backward_norm = "backward"
        elif ctx.norm == "backward":
            backward_norm = "forward"
        elif ctx.norm == "ortho":
            backward_norm = "ortho"
        else:
            raise ValueError(f"Unknown ifft normalization: {ctx.norm}")
        return (
            _cfftn(grad_output, ctx.dim, backward_norm, ctx.method),
            None,
            None,
            None,
        )


def _cfftn(x, dim, norm, method):
    if method == "shift":
        return _cfftn_shift(x, dim, norm)
    elif method == "modulate":
        return _cfftn_modulate(x, dim, norm)
    raise ValueError(f"method must be 'shift' or 'modulate', got {method!r}")


def _cifftn(x, dim, norm, method):
    if method == "shift":
        return _cifftn_shift(x, dim, norm)
    elif method == "modulate":
        return _cifftn_modulate(x, dim, norm)
    raise ValueError(f"method must be 'shift' or 'modulate', got {method!r}")


def _cfftn_shift(x, dim, norm):
    """Centered fft, shift method."""
    x = fft.ifftshift(x, dim=dim)
    x = fft.fftn(x, dim=dim, norm=norm)
    x = fft.fftshift(x, dim=dim)
    return x


def _cifftn_shift(x, dim, norm):
    """Centered ifft, shift method"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.ifftn(x, dim=dim, norm=norm)
    x = fft.fftshift(x, dim=dim)
    return x


def _cfftn_modulate(x, dim, norm):
    """Centered fft, modulate method."""
    x = x.to(_complex_dtype(x))
    if dim is None:
        dim = tuple(range(x.ndim))
    for d in dim:
        N = x.shape[d]
        phase = _fftshift_phase_ramp(
            N, mode="fftshift", dtype=_complex_dtype(x), device=x.device
        )
        x = _mul1d_at_dim(x, phase, d)
    x = fft.fftn(x, dim=dim, norm=norm)
    for d in dim:
        N = x.shape[d]
        phase = _fftshift_phase_ramp(
            N, mode="ifftshift", dtype=_complex_dtype(x), device=x.device
        )
        x = _mul1d_at_dim(x, phase, d)
    return x


def _cifftn_modulate(x, dim, norm):
    """Centered ifft, modulate method.

    The inverse DFT kernel carries the opposite sign to the forward, so the
    fold modulations are the conjugates of the cfft ones (not their negation:
    ``-exp(i*theta) != exp(-i*theta)``).
    """
    x = x.to(_complex_dtype(x))
    if dim is None:
        dim = tuple(range(x.ndim))
    for d in dim:
        N = x.shape[d]
        phase = _fftshift_phase_ramp(
            N, mode="fftshift", dtype=_complex_dtype(x), device=x.device
        )
        x = _mul1d_at_dim(x, phase.conj(), d)
    x = fft.ifftn(x, dim=dim, norm=norm)
    for d in dim:
        N = x.shape[d]
        phase = _fftshift_phase_ramp(
            N, mode="ifftshift", dtype=_complex_dtype(x), device=x.device
        )
        x = _mul1d_at_dim(x, phase.conj(), d)
    return x


_PHASE_RAMP_CACHE = {}


def _complex_dtype(x: Tensor):
    """Dtype of the result of transforming ``x``: complex64 for float32/int input,
    complex128 for float64, and unchanged for complex input. Keeping ramps at
    this dtype avoids silent float32 -> complex128 working-set upcasts."""
    if x.is_floating_point() or x.is_complex():
        return {
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
        }.get(x.dtype, torch.complex64)
    return torch.complex64


def _fftshift_phase_ramp(N: int, mode="fftshift", dtype=torch.complex128, device="cpu"):
    """Unit-modulus ramp whose multiply equals fftshift (mode="fftshift",
    applied to the input) or ifftshift (mode="ifftshift", applied to the
    output) of a length-N axis, for both even and odd N.

    Derivation (cfft, f = N // 2, torch's roll(+f)/roll(-f) conventions):
    ``fftshift(fftn(ifftshift(x)))[n] = e^{2i*pi*f*(n-f)/N} * fftn(x * e^{2i*pi*f*k/N})[n]``
    -- the (n - f) offset on the output side absorbs the constant e^{-2i*pi*f^2/N},
    which is (+/-1) for even N and genuinely complex for odd N.
    """
    key = (N, mode, dtype, device)
    if key not in _PHASE_RAMP_CACHE:
        Nover2 = N // 2
        n = torch.arange(N, dtype=torch.float64)
        if mode == "ifftshift":
            n = n - Nover2
        elif mode != "fftshift":
            raise ValueError(f"mode must be 'fftshift' or 'ifftshift', got {mode!r}")
        phase = torch.exp(2j * torch.pi * Nover2 * n / N).to(dtype)
        _PHASE_RAMP_CACHE[key] = phase.to(device)
    return _PHASE_RAMP_CACHE[key]


def _mul1d_at_dim(input_nd, input_1d, i: int):
    """Multiply the ith axis of input_nd by input_1d.
    Shapes must work out.
    Inplace multiplication for memory benefits.
    """
    # 1. Build a dynamic shape list: [1, 1, C, 1]
    # It places 1 everywhere, except at index 'i' where it places C
    broadcast_shape = [1] * input_nd.ndim
    broadcast_shape[i] = input_nd.shape[i]

    # 2. Reshape and multiply
    result = input_nd.mul_(input_1d.view(broadcast_shape))
    return result


# Convenience functions
def cfft(x: Tensor, **kwargs):
    return cfftn(x, dim=(-1,), **kwargs)


def cifft(x: Tensor, **kwargs):
    return cifftn(x, dim=(-1,), **kwargs)


def cfft2(x: Tensor, **kwargs):
    return cfftn(x, dim=(-2, -1), **kwargs)


def cifft2(x: Tensor, **kwargs):
    return cifftn(x, dim=(-2, -1), **kwargs)
