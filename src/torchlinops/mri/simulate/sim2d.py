#!/usr/bin/env python3


def gen_mri_dataset(im_size, num_coils):
    # Image
    img = sp.shepp_logan(im_size).astype(np.complex64)

    # Trajectory
    if len(im_size) == 2:
        trj = spiral_2d(im_size)
    elif len(im_size) == 3:
        trj = tgas_spi(im_size, ntr=500)
    else:
        raise ValueError(f'Unsupported image dimension: {len(im_size)} (size {im_size})')

    # Coils
    mps = sp.mri.birdcage_maps((num_coils, *im_size))
    return img, trj, mps

def get_sp_linop(im_size: Tuple, trj: np.ndarray, mps: np.ndarray):
    trj = rearrange(trj, 'K R D -> R K D')
    trj = torch.from_numpy(trj)
    # trj_tkbn = to_tkbn(trj, im_size)
    F = NUFFT(trj, im_size,
              in_batch_shape=('C',),
              out_batch_shape=('R',),
             )
    S = SENSE(torch.from_numpy(mps).to(torch.complex64))
    # R = Repeat(trj_tkbn.shape[0], dim=0, ishape=('C', 'Nx', 'Ny'), oshape=('R', 'C', 'Nx', 'Ny'))
    return F @ S

def get_gridded_linop(im_size: Tuple, trj: np.ndarray, mps: np.ndarray):
    trj = rearrange(trj, 'K R D -> R K D')
    trj = torch.from_numpy(trj)
    trj = torch.round(trj)
    F = GriddedNUFFT(
        trj, im_size,
        in_batch_shape=('C',),
        out_batch_shape=('R',),
    )
    S = SENSE(torch.from_numpy(mps).to(torch.complex64))
    return F @ S

def simulate(A: NamedLinop, img: np.ndarray, sigma: float = 0.):
    ksp = linop(torch.from_numpy(img).to(torch.complex64))
    ksp = ksp + sigma * torch.randn_like(ksp)
    return ksp
