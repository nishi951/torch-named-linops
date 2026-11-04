import pytest
import torch
from torchlinops import Dense, Chain


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
    torch.compile(lambda t: chain(t), fullgraph=True)(x)

    # If it fails, this outputs
    # torch._dynamo.exc.Unsupported: logging.Logger method not supported for non-export cases
