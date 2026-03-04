import pytest

from torchlinops.linops.device import clear_transfer_streams_registry


@pytest.fixture(autouse=True)
def clean_transfer_streams_registry():
    """Clear the transfer streams registry after each test.

    This ensures that CUDA streams don't persist between tests,
    which could cause test isolation issues.
    """
    yield
    clear_transfer_streams_registry()
