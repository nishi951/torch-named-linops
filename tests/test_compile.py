import pytest
import torch

from torchlinops import ArrayToBlocks, BlocksToArray, Chain, Dense, Identity

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton is unavailable on macOS
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _to_cast_scalar(x, out):
        value = x.to(tl.int64)
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


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
@pytest.mark.parametrize(
    "grid_size, block_size, stride",
    [
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
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
@pytest.mark.parametrize(
    "grid_size, block_size, stride",
    [
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
@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="GPU and Triton are required",
)
def test_to_cast_fails_when_scalar_arg_is_one():
    """Documents why kernels use tl.cast instead of the .to() method.

    Triton specializes scalar kernel arguments whose value is 1 as a constexpr.
    constexpr has no .to() method, so the cast inside the kernel fails during
    compilation. tl.cast handles the constexpr and materializes the target dtype.
    """
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    with pytest.raises(
        triton.compiler.errors.CompilationError, match="has no attribute 'to'"
    ):
        _to_cast_scalar[(1,)](1, out)

    # The same kernel works when the argument is not specialized.
    _to_cast_scalar[(1,)](2, out)
    assert out.item() == 2
