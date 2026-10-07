"""Helper functions"""


def scale_int(t: tuple[int, ...], scale_factor: float):
    return tuple(int(scale_factor * s) for s in t)
