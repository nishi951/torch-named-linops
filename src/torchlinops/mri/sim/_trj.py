from dataclasses import dataclass, field, asdict
from typing import Tuple

from einops import rearrange
import numpy as np
import sigpy.mri as mri

from typing import Optional

__all__ = [
    'spiral_2d',
    'tgas_spi',
    'acs',
    'radial_2d',
    'cartesian',
    'EPI_2d',
]

def spiral_2d(
        im_size: Tuple,
        n_shots: int = 16,
        alpha: float = 1.5,
        f_sampling: float = 0.4,
        g_max: float = 40.,
        s_max: float = 100.,
) -> np.ndarray:
    """
    Generates an 2-dimensional variable density spiral

    Parameters:
    ----------
    im_size: Tuple
        2D Image resolution tuple
    n_shots : int
        number of phase encodes to cover k-space
    alpha : float
        controls variable density. 1.0 means no variable density, center denisty increases with alpha
    g_max : float
        Maximum gradient amplitude in T/m
    s_max : float
        Maximum gradient slew rate in T/m/s

    Returns:
    ----------
    trj : np.ndarray <float>
        k-space trajector with shape (n_shots, n_readout_points,  d), d = len(self.im_size)
    """

    # Gen spiral
    trj = mri.spiral(
        fov=1,
        N=max(im_size),
        f_sampling=f_sampling, # TODO function of self.n_read
        R=1,
        ninterleaves=n_shots,
        alpha=alpha,
        gm=g_max, # Tesla / m
        sm=s_max, # Tesla / m / s
    )
    assert trj.shape[0] % n_shots == 0
    trj = trj.reshape((trj.shape[0] // n_shots, n_shots, 2), order='F')

    # Equalize axes
    for i in range(trj.shape[-1]):
        trj[..., i] *= im_size[i] / 2 / trj[..., i].max()

    return trj


def rotation_matrix(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)

    return np.array(
        [
            [
                a * a + b * b - c * c - d * d,
                2 * (b * c - a * d),
                2 * (b * d + a * c),
            ],
            [
                2 * (b * c + a * d),
                a * a + c * c - b * b - d * d,
                2 * (c * d - a * b),
            ],
            [
                2 * (b * d - a * c),
                2 * (c * d + a * b),
                a * a + d * d - b * b - c * c,
            ],
        ]
    )


def tgas_spi(
        im_size,
        ntr: int,
        n_shots: Optional[int] = 16,
        R: Optional[float] = 1) -> np.ndarray:
    """
    Generates a sample k-space trajectory for MRF, after the following paper:
    Optimized multi-axis spiral projection <scp>MR</scp> fingerprinting with
    subspace reconstruction for rapid whole-brain high-isotropic-resolution quantitative imaging
    Cao, X. et al (2022), https://doi.org/10.1002/mrm.29194

    Parameters
    ----------
    ntr : int
        number of TRs in MRF sequence
    R : float
        the spatial undersampling factor
    n_shots : int
        number of interleaves needed to cover 2D k-space for one spiral

    Returns
    ----------
    trj : torch.tensor <float>
        k-space trajectory with shape (nro, npe, ntr, d), npe=ngroups
    """

    # Consts
    d = len(im_size)

    # Generate base spiral
    base_spiral = spiral_2d(im_size, n_shots)

    # Tiny golden angle
    tga = np.deg2rad(22.63)

    # Rotation matrix

    # 2D Case
    if d == 2:

        # Get rotation axis
        axis = np.zeros(3)
        axis[-1] = 1

        # Return trajectory
        trj = np.zeros((base_spiral.shape[0], n_shots, ntr, d))

        # Randomize spirals
        for i in range(n_shots):
            for j in range(ntr):

                # TGA along interleave dim
                theta = tga * (i + j)
                rot = rotation_matrix(axis, theta)[:2, :2]
                trj[:, i, j, :] = base_spiral[:, 0, :] @ rot

        # undersample groups
        ngroups_undersamp = round(trj.shape[1] / R)
        trj = trj[:, :ngroups_undersamp, ...]

    else:

        # Trajectory dimensions
        n_inter_undersamp = round(n_shots / R)
        ngroups = n_inter_undersamp * 3
        nro = base_spiral.shape[0]
        trj = np.zeros((nro, ngroups, ntr, 3))

        # Rotate base spiral in all three dimensions
        for dim in range(3):

            # Get in plane spiral
            ax1 = dim
            ax2 = (dim + 1) % 3
            in_plane_spiral = np.zeros((*base_spiral.shape[:-1], 3))
            in_plane_spiral[..., ax1] = base_spiral[..., 0]
            in_plane_spiral[..., ax2] = base_spiral[..., 1]

            # Set axis of rotation
            rotation_axis = np.zeros(3)
            rotation_axis[dim] = 1

            # Rotations by TGA
            for j in range(ntr):

                # Randomly select group indices for subsampling
                random_groups = np.random.choice(n_shots, n_inter_undersamp)
                for i, g in enumerate(random_groups):

                    # Get rotation matrix
                    theta = tga * (g + j)
                    rot = rotation_matrix(rotation_axis, theta)

                    # apply to correct group
                    grp = dim * n_inter_undersamp + i
                    trj[:, grp, j, :] = in_plane_spiral[:, g, :] @ rot

    return trj


def radial_2d(im_size: Tuple, n_read: Optional[int] = None, n_shots: Optional[int] = None) -> np.ndarray:
    """
    Generates a 2d radial trajectory

    Parameters:
    -----------
    n_read : int
        number of readout points, defaults to 2X oversampling

    Returns:
    ----------
    trj : np.ndarray <float>
        k-space trajector with shape (n_read, n_shots_rad, d), d = len(self.im_size)
    """
    N = max(im_size)
    n_shots = n_shots if n_shots is not None else 2*N

    # Number of spokes for fully sampled
    n_spokes = round(np.pi * N)
    thetas = np.linspace(0, np.pi, n_spokes, endpoint=False)[:, None] + 1e-5 # numerical reasons ...

    # Generate points along the readout
    readout_line = np.linspace(-N/2, N/2, n_read)[None, :]

    # Rotations
    pts = readout_line * np.exp(1j * thetas)

    # gen k-space
    trj = np.array([pts.real, pts.imag]).T

    # Rescale
    for i in range(trj.shape[-1]):
        trj[..., i] = trj[..., i] * self.im_size[i] / trj[..., i].max() / 2

    return trj

def cartesian(im_size, n_read: Optional[int] = None):
    """
    Generates a cartesian trajectory

    Parameters:
    -----------
    n_read : int
        number of readout points, defaults to 1X oversampling

    Returns:
    ----------
    trj : np.ndarray <float>
        k-space trajector with shape (n_read, nky, nkz, ..., d), d = len(self.im_size)
    """
    # Resample longest axis by n_read
    i_max = np.argmax(im_size)

    # Defualt number of readouts
    if n_read is None:
        n_read = im_size[i_max]

    # Make trajectory
    new_im_size = list(im_size)
    new_im_size[i_max] = n_read
    trj = np.zeros((*new_im_size, len(new_im_size)))
    d = trj.shape[-1]
    for i in range(d):

        # Get number of points
        n_read = new_im_size[i]
        n = im_size[i]

        # Update trajectory
        tup = (None,) * i + (slice(None),) + (None,) * (d - i - 1)
        trj[..., i] = np.linspace(-n/2, n/2, n_read)[tup]

    # Move axes and flatten
    trj = np.moveaxis(trj, i_max, 0)

    return trj

def EPI_2d(im_size
            n_read: Optional[int] = None,
            n_shots: Optional[int] = 16) -> np.ndarray:
    """
    Generates a cartesian trajectory

    Parameters:
    -----------
    n_read : int
        number of readout points, defaults to 1X oversampling
    n_shots : int
        number of phase encodes to cover k-space

    Returns:
    ----------
    trj : np.ndarray <float>
        k-space trajector with shape (n_read, n_shots, d), d = len(self.im_size)
    """

    # Start with cartesian trajectory
    d = len(im_size)
    trj_cart = cartesian(n_read)
    assert trj_cart.shape[0] == n_read
    assert trj_cart.shape[-1] == d
    assert len(trj_cart.shape) == 3

    # Make sure number of shots divides nicely
    nky = trj_cart.shape[1]
    assert nky % n_shots == 0, 'Number of shots should be a divisor of number of acquired ky lines'
    lines_per_shot = nky // n_shots

    # Place holder trajecrory
    n_read_shot = n_read * lines_per_shot
    trj = np.zeros((n_read * lines_per_shot, n_shots, 2))

    # Populate
    for i in range(n_shots):

        # Flip lines
        lines = trj_cart[:, i::n_shots, :]
        lines[:, ::2, :] = np.flip(lines[:, ::2, :], axis=0)

        # Update trj
        trj[:, i, :] = lines.reshape((-1, d), order='F')

    return trj

def acs(calib_size: tuple) -> np.ndarray:
    """
    Generates a calibration trajectory for things like Espirit and GRAPPA

    Parameters:
    -----------
    calib_size : tuple
        dimension of calibration as tuple of ints

    Returns:
    ----------
    trj : np.ndarray <float>
        k-space trajector with shape (*calib_size, d), d = len(calib_size)
    """

    # Make trajectory
    d = len(calib_size)
    trj = np.zeros((*calib_size, d))

    # Set values
    for i in range(d):
        n  = calib_size[i]
        tup = (None,) * i + (slice(None),) + (None,) * (d - i - 1)
        trj[..., i] = np.linspace(-n/2, n/2, n)[tup]

    return trj
