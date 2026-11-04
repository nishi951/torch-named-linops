import pytest
import torch
from torchlinops import Chain, Dense, Identity


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
