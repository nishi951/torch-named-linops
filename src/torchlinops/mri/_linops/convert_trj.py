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


def convert_trj(trj, from_type, to_type, im_size=None):
    """Convert between various trajectory types and normalizations
    sigpy: range [-N//2, N//2], shape [... K D]
    tkbn:  range [-pi, pi], shape [... D K]
    mat: range[-1/2, 1/2], shape[... K D]
    """
    if (from_type == "sigpy" or to_type == "sigpy") and im_size == None:
        raise ValueError("Must specify im_size if converting to/from sigpy.")
    if from_type == to_type:
        return trj
    # Convert to mat first
    if from_type == "mat":
        pass
    elif from_type == "tkbn":
        trj = trj / (2 * np.pi)
        trj = rearrange(trj, "... d k -> ... k d")
    elif from_type == "sigpy":
        for i, N in enumerate(im_size):
            trj[..., i] = trj[..., i] / N
    else:
        raise ValueError(f"Invalid from_type: {from_type}")
    # Convert to whatever else
    if to_type == "mat":
        pass
    elif to_type == "tkbn":
        trj = trj * 2 * np.pi
        trj = rearrange(trj, "... k d -> ... d k")
    elif to_type == "sigpy":
        for i, N in enumerate(im_size):
            trj[..., i] = trj[..., i] * N
    else:
        raise ValueError(f"Invalid to_type: {to_type}")
    return trj
