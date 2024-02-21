from typing import Optional, Tuple

import torch
import torch.fft as fft
import sigpy as sp
import sigpy.mri as mri
import numpy as np

from torchlinops.core.linops import Diagonal
from torchlinops.mri.linops import NUFFT
from torchlinops.mri.recon.pcg import CGHparams, ConjugateGradient

def sp_fft(x, dim=None):
    """Matches Sigpy's fft, but in torch"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.fftn(x, dim=dim, norm='ortho')
    x = fft.fftshift(x, dim=dim)
    return x

def sp_ifft(x, dim=None, norm=None):
    """Matches Sigpy's fft adjoint, but in torch"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.ifftn(x, dim=dim, norm='ortho')
    x = fft.fftshift(x, dim=dim)
    return x

def truncate_trj_ksp(trj, ksp, max_k, dcf: Optional[np.ndarray] = None):
    mask = np.all(np.abs(trj) <= max_k, axis=-1)
    trj_truncated = trj[mask, :] # [B... K D] -> [K' D]
    ksp_truncated = ksp[:, mask] # [C B... K] -> [C K']
    if dcf is not None:
        dcf_truncated = dcf[mask] # [B... K] -> [K']
        return trj_truncated, ksp_truncated, dcf_truncated
    return trj_truncated, ksp_truncated

def inufft(
    trj: torch.Tensor,
    ksp: torch.Tensor,
    im_size: Tuple,
    dcf: Optional[torch.Tensor] = None,
    num_iter=10,
    device='cpu',
):
    """Inverse NUFFT aka least squares via PCG
    trj: [B... K D] sigpy-style trajectory
    ksp: [C B... K] tensor where B... is the same batch as omega's B...
    dcf: if provided, [B... K]
    """
    hparams = CGHparams(num_iter=num_iter)
    device = ksp.device
    C = ksp.shape[0]
    batch = tuple(f'B{i}' for i in range(len(ksp.shape[1:-1])))
    # Create simple linop
    F = NUFFT(trj, im_size,
              in_batch_shape=('C',),
              out_batch_shape=batch).to(device)
    if dcf is not None:
        D = Diagonal(torch.sqrt(dcf),
                     ioshape=('C', *batch, 'K'),
                    ).to(device)
    else:
        D = Identity(ioshape=('C', *batch, 'K')).to(device)
    A = D @ F
    AHb = A.H(D(ksp))
    cg = ConjugateGradient(A.N, hparams).to(device)
    return cg(AHb, AHb)

def synth_cal(
    trj,
    ksp,
    acs_size: int,
    dcf: Optional[np.ndarray] = None,
    device: torch.device = 'cpu',
):
    D = trj.shape[-1]
    cal_size = (acs_size,) * D
    if dcf is not None:
        trj, ksp, dcf = truncate_trj_ksp(trj, ksp, max_k=acs_size/2, dcf=dcf)
    else:
        trj, ksp = truncate_trj_ksp(trj, ksp, max_k=acs_size/2)
    # trj = rearrange(trj, 'K R D -> R K D')
    trj = torch.from_numpy(trj)
    #omega = to_tkbn(trj, cal_size)
    ksp = torch.from_numpy(ksp).to(torch.complex64)
    if dcf is not None:
        dcf = torch.as_tensor(dcf)
    img_cal = inufft(trj, ksp, cal_size, dcf=dcf, num_iter=10, device=device)
    kgrid = sp_fft(img_cal, dim=(-2, -1))
    # kgrid = fft.fftn(img_cal, dim=(-2, -1))
    return kgrid.detach().cpu().numpy()

def get_mps_kgrid(trj, ksp, im_size, calib_width, kernel_width, device_idx, **espirit_kwargs):
    """
    """
    if len(espirit_kwargs) == 0:
        # Defaults
        espirit_kwargs = {
            'crop': 0.8,
            'thresh': 0.05,
        }
    device = torch.device(f'cuda:{device_idx}' if device_idx >= 0 else 'cpu')
    dcf = mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx), show_pbar=False)
    xp = sp.get_device(dcf).xp
    dcf /= xp.linalg.norm(dcf)
    kgrid = synth_cal(trj, ksp, calib_width, dcf, device)
    kgrid_pad = sp.resize(kgrid, (kgrid.shape[0], *im_size))
    mps = mri.app.EspiritCalib(
        kgrid_pad,
        calib_width=calib_width,
        kernel_width=kernel_width,
        device=sp.Device(device_idx),
        **espirit_kwargs,
    ).run()
    return sp.to_device(mps, sp.cpu_device), kgrid
