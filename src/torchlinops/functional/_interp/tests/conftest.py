import pytest


@pytest.fixture
def small1d():
    N = (2,)
    grid_size = (10,)
    # npts = 20
    locs_batch_size = (20,)
    width = 4.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec


@pytest.fixture
def medium1d():
    N = (2, 2)
    grid_size = (24,)
    locs_batch_size = (6, 11)
    width = 4.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec


@pytest.fixture
def large1d():
    N = (2,)
    grid_size = (220,)
    locs_batch_size = (8, 8, 4)
    width = 4.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec


@pytest.fixture
def tiny2d():
    N = (2,)
    grid_size = (2, 2)
    locs_batch_size = (1,)
    width = 2.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec


@pytest.fixture
def small2d():
    N = (2, 2)
    grid_size = (10, 8)
    locs_batch_size = (9,)
    width = 4.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec


@pytest.fixture
def medium2d():
    N = (2, 2)
    grid_size = (124, 125)
    locs_batch_size = (3, 19)
    # Kernel size larger than 4 might cause test to fail
    # because of numerical issues
    width = 4.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec


@pytest.fixture
def large2d():
    N = (2, 2)
    grid_size = (220, 220)
    locs_batch_size = (4, 3, 10)
    width = 4.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec


@pytest.fixture
def small3d():
    N = (1, 1)
    grid_size = (10, 8, 9)
    locs_batch_size = (3, 5)
    width = 4.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec


@pytest.fixture
def medium3d():
    N = (2, 2)
    grid_size = (33, 35, 37)
    locs_batch_size = (3, 19)
    width = 4.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec


@pytest.fixture
def large3d():
    N = (1, 1)
    grid_size = (220, 220, 220)
    locs_batch_size = (48, 500, 1600)
    width = 4.0

    spec = {
        "N": N,
        "grid_size": grid_size,
        "locs_batch_size": locs_batch_size,
        "width": width,
    }
    return spec
