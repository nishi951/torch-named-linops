from torchlinops.utils import ceildiv

from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)
from torchlinops.mri.sim.tgas_spi import (
    TGASSPISimulator,
    TGASSPISimulatorConfig,
)
from torchlinops.mri._linops.nufft import NUFFT
from torchlinops.mri._linops.nufft.timeseg import timeseg


def test_spiral2d_timeseg():
    config = Spiral2dSimulatorConfig(
        im_size=(64, 64),
        num_coils=8,
        noise_std=0.1,
        spiral_2d_kwargs={
            "n_shots": 16,
            "alpha": 1.5,
            "f_sampling": 1.0,
        },
    )
    num_segments = 5

    simulator = Spiral2dSimulator(config)
    data = simulator.data

    F = NUFFT(
        data.trj,
        im_size=config.im_size,
        in_batch_shape=("C",),
        out_batch_shape=("R", "K"),
        backend="fi",
    )
    Fseg = timeseg(F, num_segments, "B")

    assert Fseg[-1].trj.shape[-2] == ceildiv(data.trj.shape[-2], num_segments)
    assert Fseg.ishape == ("B", "C", "Nx", "Ny")
    assert Fseg.oshape == ("C", "R", "K")


def test_tgasspi_timeseg():
    config = TGASSPISimulatorConfig(
        im_size=(64, 64, 64),
        num_coils=8,
        num_TRs=100,
        num_groups=18,
        groups_undersamp=1,
        noise_std=0.1,
        spiral_2d_kwargs={
            "alpha": 1.5,
            "f_sampling": 1.0,
        },
    )
    num_segments = 5

    simulator = TGASSPISimulator(config)
    data = simulator.data

    F = NUFFT(
        data.trj,
        im_size=config.im_size,
        in_batch_shape=("C",),
        out_batch_shape=("R", "T", "K"),
        backend="fi",
    )
    Fseg = timeseg(F, num_segments, "B")

    assert Fseg[-1].trj.shape[-2] == ceildiv(data.trj.shape[-2], num_segments)
    assert Fseg.ishape == ("B", "C", "Nx", "Ny", "Nz")
    assert Fseg.oshape == ("C", "R", "T", "K")
