from typing import Optional

import numpy as np
from sigpy.sim import rotation_matrix, sl_scales, sl_offsets, sl_angles

__all__ = ["quantitative_shepp_logan"]

QSL_PD = [1.0, 0.97, 0.98, 0.98, 0.97, 0.96, 0.96, 0.96, 0.96, 0.96]  # Arbitrary units
QSL_T1 = [
    371.0,  # Fat ring
    800.0,  # White matter
    4000.0,  # CSF
    4000.0,  # CSF
    1193.0,  # Gray matter
    1193.0,  # Gray matter
    1000.0,  # Gray matter
    1202.0,  # Gray matter
    1242.0,  # Gray matter
    1176.0,  # Gray matter
]  # ms
QSL_T2 = [
    68.0,  # Fat
    75.0,  # White matter
    767.0,  # CSF
    767.0,  # CSF
    109.0,  # Gray matter
    109.0,  # Gray matter
    98.0,  # Gray matter
    79.0,  # Gray matter (different)
    89.0,  # Gray matter (different)
    100.0,  # Gray matter (different)
]  # ms


def phantom_inplace(shape, amps, scales, offsets, angles, dtype):
    """
    Generate a cube of given shape using a list of ellipsoid
    parameters.
    """

    if len(shape) == 2:
        ndim = 2
        shape = (1, shape[-2], shape[-1])

    elif len(shape) == 3:
        ndim = 3

    else:
        raise ValueError("Incorrect dimension")

    out = np.zeros(shape, dtype=dtype)

    z, y, x = np.mgrid[
        -(shape[-3] // 2) : ((shape[-3] + 1) // 2),
        -(shape[-2] // 2) : ((shape[-2] + 1) // 2),
        -(shape[-1] // 2) : ((shape[-1] + 1) // 2),
    ]

    coords = np.stack(
        (
            x.ravel() / shape[-1] * 2,
            y.ravel() / shape[-2] * 2,
            z.ravel() / shape[-3] * 2,
        )
    )

    for amp, scale, offset, angle in zip(amps, scales, offsets, angles):
        ellipsoid_inplace(amp, scale, offset, angle, coords, out)

    if ndim == 2:
        return out[0, :, :]

    else:
        return out


def ellipsoid_inplace(amp, scale, offset, angle, coords, out):
    """
    Generate a cube containing an ellipsoid defined by its parameters.
    If out is given, fills the given cube instead of creating a new
    one.
    """
    R = rotation_matrix(angle)
    coords = (np.matmul(R, coords) - np.reshape(offset, (3, 1))) / np.reshape(
        scale, (3, 1)
    )

    r2 = np.sum(coords**2, axis=0).reshape(out.shape)

    # Assign rather than accumulate
    out[r2 <= 1] = amp


def quantitative_shepp_logan(
    im_size,
    pd_vals: Optional[list] = None,
    t1_vals: Optional[list] = None,
    t2_vals: Optional[list] = None,
    dtype=float,
):
    pd_vals = pd_vals if pd_vals is not None else QSL_PD
    t1_vals = t1_vals if t1_vals is not None else QSL_T1
    t2_vals = t2_vals if t2_vals is not None else QSL_T2
    pd = phantom_inplace(im_size, pd_vals, sl_scales, sl_offsets, sl_angles, dtype)
    t1 = phantom_inplace(im_size, t1_vals, sl_scales, sl_offsets, sl_angles, dtype)
    t2 = phantom_inplace(im_size, t2_vals, sl_scales, sl_offsets, sl_angles, dtype)
    return pd, t1, t2
