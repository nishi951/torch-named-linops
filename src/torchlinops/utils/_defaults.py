__all__ = ["default_to", "default_to_dict"]


def default_to(*vals, typecast: bool = False):
    """Get the first non-None value, right to left order.

    Most "default" value goes first.
    """
    if len(vals) == 0:
        return None
    typecls = type(vals[0])
    if len(vals) == 1:
        return vals[0]
    for val in reversed(vals):
        if val is not None:
            if typecast:
                return typecls(val)
            return val


def default_to_dict(*dicts) -> dict:
    """Update a dict with default values progressively from left to right."""
    if len(dicts) == 0:
        return {}
    out = {}
    for d in dicts:
        if d is None:
            pass
        elif not isinstance(d, dict):
            raise ValueError(
                f"Non-dictionary found during default dictionary creation: {d}"
            )
        else:
            out.update(d)
    return out
