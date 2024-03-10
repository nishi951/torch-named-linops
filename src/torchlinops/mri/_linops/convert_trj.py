from math import pi

__all__ = [
    "sp2fi",
    "fi2sp",
]


def sp2fi(trj, im_size):
    """Convert from a trajectory in []"""
    assert len(im_size) == trj.shape[-1]
    for i in range(len(im_size)):
        trj[..., i] *= 2 * pi / im_size[i]
    return trj


def fi2sp(trj, im_size):
    assert len(im_size) == trj.shape[-1]
    for i in range(len(im_size)):
        trj[..., i] *= im_size[i] / (2 * pi)
    return trj


