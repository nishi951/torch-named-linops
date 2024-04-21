import logging
from time import perf_counter
from typing import Tuple, Optional

from einops import rearrange
import torch
from tqdm import tqdm

from torchlinops import Dense, get2dor3d
from torchlinops.mri.recon import PowerMethod
from torchlinops.mri.recon.prior.block import Block
from torchlinops.utils import cifft


logger = logging.getLogger(__name__)

__all__ = ["espirit"]


def espirit(
    ksp_cal: torch.Tensor,
    im_size: Tuple,
    thresh: Optional[float] = 0.02,
    kernel_width: Optional[int] = 6,
    crop: Optional[float] = 0.95,
    max_iter: Optional[int] = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Original code from mr_recon (Daniel Abraham 2024)

    Copy of sigpy implementation of ESPIRiT calibration, but in torch:
    Martin Uecker, ... ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI

    Parameters:
    -----------
    ksp_cal : torch.Tensor
        Calibration k-space data with shape (ncoil, *cal_size)
    im_size : tuple
        output image size
    thresh : float
        threshold for SVD nullspace
    kernel_width : int
        width of calibration kernel
    crop : float
        output mask based on copping eignevalues
    max_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar

    Returns:
    --------
    mps : torch.Tensor
        coil sensitivity maps with shape (ncoil, *im_size)
    eigen_vals : torch.Tensor
        eigenvalues with shape (*im_size)
    """

    # Consts
    img_ndim = len(im_size)
    num_coils = ksp_cal.shape[0]
    device = ksp_cal.device

    # Get calibration matrix.
    # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
    block_size = tuple([kernel_width] * img_ndim)
    block_stride = tuple([1] * img_ndim)
    block = Block(block_size, block_stride, input_type=ksp_cal.dtype)
    mat = block(ksp_cal)
    mat = rearrange(mat, "nc nb ... -> nb (nc ...)")

    # Perform SVD on calibration matrix
    start = perf_counter()
    logger.info("Computing SVD on calibration matrix")
    _, S, VH = torch.linalg.svd(mat, full_matrices=False)
    VH = VH[S > thresh * S.max(), :]
    total = perf_counter() - start
    logger.info(f"SVD Time: {total}")

    # Get kernels
    num_kernels = len(VH)
    kernels = VH.reshape([num_kernels, num_coils] + [kernel_width] * img_ndim)

    # Get covariance matrix in image domain
    AHA = torch.zeros(
        im_size + (num_coils, num_coils), dtype=ksp_cal.dtype, device=device
    )
    for kernel in tqdm(kernels, desc="Building espirit matrix"):
        img_kernel = cifft(
            kernel, oshape=(num_coils, *im_size), dim=tuple(range(-img_ndim, 0))
        )
        aH = rearrange(img_kernel, "nc ... -> ... nc 1")
        a = aH.swapaxes(-1, -2).conj()
        AHA += aH @ a
    AHA *= torch.prod(torch.tensor(im_size)).item() / kernel_width**img_ndim
    im_shape = get2dor3d(im_size)
    # TODO: Maybe make the A matrix earlier?
    AHA = Dense(
        AHA,
        weightshape=(*im_shape, "C", "C1"),
        ishape=(*im_shape, "C"),
        oshape=(*im_shape, "C1"),
    )
    ishape = tuple(AHA.size(d) for d in AHA.ishape)

    # Get eigenvalues and eigenvectors
    power_method = PowerMethod(num_iter=max_iter)
    mps, eigen_vals = power_method(AHA, ishape=ishape, device=device)
    # mps, eigen_vals = power_method_matrix(AHA, num_iter=max_iter, verbose=verbose)

    # Phase relative to first map
    mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-8))

    # Crop
    mps *= eigen_vals > crop

    return mps, eigen_vals
