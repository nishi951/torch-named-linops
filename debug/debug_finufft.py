import cufinufft
import torch

from torchlinops.utils.benchmark import torch_benchmark

# number of nonuniform points
nro = 2000 * 500 * 32
ncoil = 12

# grid size
im_size = (220, 220, 220)

torch_dev = torch.device(0)

# generate positions for the nonuniform points and the coefficients
trj_gpu = 2 * torch.pi * torch.rand(size=(nro, 3)).to(torch_dev)
ksp_gpu = (torch.randn(size=(ncoil, nro)) + 1j * torch.randn(size=(ncoil, nro))).to(
    torch_dev
)


def trj2contig(trj):
    return (
        trj_gpu[..., 0].contiguous(),
        trj_gpu[..., 1].contiguous(),
        trj_gpu[..., 2].contiguous(),
    )


def run(trj_gpu, ksp_gpu):
    # compute the transform
    f_gpu = cufinufft.nufft3d1(
        *trj2contig(trj_gpu),
        ksp_gpu,
        im_size,
    )


# timing, memory = torch_benchmark(run, n_trials=5, warmup=True, trj_gpu=trj_gpu, ksp_gpu=ksp_gpu)
# breakpoint()

# Try planned version
plan = cufinufft.Plan(1, im_size, n_trans=ncoil)
plan.setpts(*trj2contig(trj_gpu))


def run_fast(ksp_gpu):
    # compute the transform
    plan.execute(ksp_gpu)


timing_fast, memory_fast = torch_benchmark(
    run_fast, n_trials=5, warmup=True, ksp_gpu=ksp_gpu
)
breakpoint()
