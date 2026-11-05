import pytest
import torch

from torchlinops import NUFFT, ArrayToBlocks, BlocksToArray, Chain, Dense, Identity
from torchlinops.functional import interpolate, interpolate_adjoint
from torchlinops.functional._interp import grid, ungrid
from torchlinops import (
    NUFFT,
    ArrayToBlocks,
    BlocksToArray,
    Chain,
    Concat,
    Dense,
    Identity,
    Stack,
)
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton is unavailable on macOS
    triton = None
    tl = None


TRITON_GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="GPU and Triton are required",
)


if triton is not None:

    @triton.jit
    def _to_cast_scalar(x, out):
        value = x.to(tl.int64)
        tl.store(out, value)

    @triton.jit
    def _tl_cast_scalar(x, out):
        value = tl.cast(x, tl.int64)
        tl.store(out, value)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_dense_chain_compiles():
    W1 = torch.randn(8, 16, device="cuda")
    W2 = torch.randn(8, 8, device="cuda")
    chain = Chain(
        Dense(W1, ("M", "N"), ("N",), ("M",)),
        Dense(W2, ("M", "M"), ("M",), ("M",)),
    )
    x = torch.randn(16, device="cuda")

    compiled = torch.compile(lambda t: chain(t), fullgraph=True)
    out = compiled(x)

    torch.testing.assert_close(out, chain(x))
    # If it fails, this outputs
    # torch._dynamo.exc.Unsupported: logging.Logger method not supported for non-export cases


def test_identity_adjoint_compiles():
    """CPU coverage for the `H` debug-log guard; runs in CI without a GPU."""
    linop = Identity(("N",))
    x = torch.randn(16)

    compiled = torch.compile(lambda t: linop.H(t), fullgraph=True)
    out = compiled(x)

    torch.testing.assert_close(out, linop.H(x))


def test_concat_dense_compiles_cpu():
    """Regression coverage for #230: Concat traces Dense.size during compile."""
    A = Dense(
        torch.randn(1, 3, 4, dtype=torch.complex64),
        ("N", "P", "Q"),
        ("N", "Q"),
        ("N", "P"),
    )
    B = Dense(
        torch.randn(1, 3, 4, dtype=torch.complex64),
        ("N", "P", "Q"),
        ("N", "Q"),
        ("N", "P"),
    )
    linop = Concat(A, B, idim="Q")
    x = torch.randn(1, 8, dtype=torch.complex64)

    compiled = torch.compile(lambda t: linop(t))

    torch.testing.assert_close(compiled(x), linop(x))


def test_stack_dense_compiles_cpu():
    """Regression coverage for #230: Stack traces Dense.size during compile."""
    A = Dense(
        torch.randn(1, 3, 4, dtype=torch.complex64),
        ("N", "P", "Q"),
        ("N", "Q"),
        ("N", "P"),
    )
    B = Dense(
        torch.randn(1, 3, 4, dtype=torch.complex64),
        ("N", "P", "Q"),
        ("N", "Q"),
        ("N", "P"),
    )
    linop = Stack(A, B, odim_and_idx=("Q", 1))
    x = torch.randn(1, 4, dtype=torch.complex64)

    compiled = torch.compile(lambda t: linop(t))

    torch.testing.assert_close(compiled(x), linop(x))


def test_dense_multi_axis_ellipsis_size_compiles():
    """Dense.size must resolve axes after a multi-axis ellipsis under Dynamo."""
    weight = torch.randn(2, 7, 5, 3)
    linop = Dense(weight, ("...", "M", "N"), ("...", "M"), ("...", "N"))
    x = torch.randn(4)

    def fn(t):
        return t + linop.size("M") + linop.size("N")

    compiled = torch.compile(fn, fullgraph=True)

    torch.testing.assert_close(compiled(x), x + 5 + 3)


@pytest.mark.gpu
@TRITON_GPU_REQUIRED
@pytest.mark.parametrize(
    "grid_size, block_size, stride",
    [
        pytest.param((8,), (4,), (4,), id="1d-regular"),
        pytest.param((1,), (1,), (1,), id="1d-size-1"),
        pytest.param((8, 8), (4, 4), (4, 4), id="regular"),
        # Triton specializes scalar kernel arguments equal to 1. Exercise that
        # path so the scalar_cast workaround is not lost under torch.compile.
        pytest.param((1, 8, 8), (1, 4, 4), (1, 4, 4), id="size-1"),
    ],
)
def test_compiled_array_to_blocks_matches_eager(grid_size, block_size, stride):
    op = ArrayToBlocks(grid_size, block_size, stride)
    x = torch.randn(2, *grid_size, dtype=torch.complex64, device="cuda")

    torch.testing.assert_close(torch.compile(op)(x), op(x))


@pytest.mark.gpu
@TRITON_GPU_REQUIRED
@pytest.mark.parametrize(
    "grid_size, block_size, stride",
    [
        pytest.param((8,), (4,), (4,), id="1d-regular"),
        pytest.param((1,), (1,), (1,), id="1d-size-1"),
        pytest.param((8, 8), (4, 4), (4, 4), id="regular"),
        pytest.param((1, 8, 8), (1, 4, 4), (1, 4, 4), id="size-1"),
    ],
)
def test_compiled_blocks_to_array_matches_eager(grid_size, block_size, stride):
    a2b = ArrayToBlocks(grid_size, block_size, stride)
    op = BlocksToArray(grid_size, block_size, stride)
    x = torch.randn(2, *grid_size, dtype=torch.float32, device="cuda")
    blocks = a2b(x)

    torch.testing.assert_close(torch.compile(op)(blocks), op(blocks))


@pytest.mark.gpu
@TRITON_GPU_REQUIRED
def test_to_cast_fails_when_scalar_arg_is_one():
    """Documents why kernels use tl.cast instead of the .to() method.

    Triton specializes scalar kernel arguments whose value is 1 as a constexpr.
    constexpr has no .to() method, so the cast inside the kernel fails during
    compilation. tl.cast handles the constexpr and materializes the target dtype.
    """
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    with pytest.raises(triton.compiler.errors.CompilationError, match="to"):
        _to_cast_scalar[(1,)](1, out)

    # The same .to() kernel works when the argument is not specialized.
    _to_cast_scalar[(1,)](2, out)
    assert out.item() == 2

    # tl.cast handles both the specialized value and a normal value.
    _tl_cast_scalar[(1,)](1, out)
    assert out.item() == 1
    _tl_cast_scalar[(1,)](2, out)
    assert out.item() == 2


@pytest.mark.gpu
@TRITON_GPU_REQUIRED
@pytest.mark.parametrize("op", ["grid", "ungrid"])
def test_functional_interp_complex_compiles(op):
    locs = get_valid_locs((8,), (16, 16), 2, 4.0, "cpu", centered=True).cuda()
    if op == "ungrid":
        x = torch.randn(16, 16, dtype=torch.complex64, device="cuda")
        fn = lambda t: ungrid(t, locs, 4.0)
    else:
        x = torch.randn(8, dtype=torch.complex64, device="cuda")
        fn = lambda t: grid(t, locs, (16, 16), 4.0)

    compiled = torch.compile(fn, fullgraph=True)
    torch.testing.assert_close(compiled(x), fn(x))


@pytest.mark.gpu
@TRITON_GPU_REQUIRED
@pytest.mark.parametrize("mode", ["forward", "adjoint"])
def test_interp_locs_grad_does_not_propagate_compiled(mode):
    locs = get_valid_locs((8,), (16, 16), 2, 4.0, "cpu", centered=True).cuda()
    locs.requires_grad_(True)

    if mode == "forward":
        x = torch.randn(16, 16, dtype=torch.complex64, device="cuda")
        fn = lambda t: interpolate(t, locs, 4.0)
    else:
        x = torch.randn(8, dtype=torch.complex64, device="cuda")
        fn = lambda t: interpolate_adjoint(t, locs, (16, 16), 4.0)

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(x)
    assert out.requires_grad is False
    torch.testing.assert_close(out, fn(x))


@pytest.mark.gpu
@TRITON_GPU_REQUIRED
def test_nufft_complex_compiles_forward_and_backward():
    locs = get_valid_locs((8,), (16, 16), 2, 4.0, "cpu", centered=True).cuda()
    op = NUFFT(locs.clone(), (16, 16), output_shape=("R",), width=4.0, oversamp=1.25)

    x = torch.randn(16, 16, dtype=torch.complex64, device="cuda")
    compiled = torch.compile(lambda t: op(t), fullgraph=True)
    torch.testing.assert_close(compiled(x), op(x))

    x_compiled = x.clone().requires_grad_(True)
    x_eager = x.clone().requires_grad_(True)
    compiled(x_compiled).abs().sum().backward()
    op(x_eager).abs().sum().backward()
    torch.testing.assert_close(x_compiled.grad, x_eager.grad)
