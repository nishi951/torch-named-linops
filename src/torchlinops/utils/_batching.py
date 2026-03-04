import itertools

from tqdm import tqdm

__all__ = ["ceildiv", "batch_iterator", "batch_tqdm", "dict_product"]


def ceildiv(dividend, divisor):
    """Ceiling division -- return ceil(a/b) as an integer.

    Parameters
    ----------
    dividend : int
        The numerator.
    divisor : int
        The denominator.

    Returns
    -------
    int
        ``ceil(dividend / divisor)``.
    """
    return -(-dividend // divisor)


def batch_iterator(total, batch_size):
    """Yield ``(start, end)`` index pairs for iterating over data in batches.

    Parameters
    ----------
    total : int
        Total number of elements.
    batch_size : int
        Maximum number of elements per batch.

    Yields
    ------
    tuple of (int, int)
        ``(start, end)`` indices for each batch.
    """
    assert total > 0, f"batch_iterator called with {total} elements"
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])


def batch_tqdm(total, batch_size, **tqdm_kwargs):
    """Like :func:`batch_iterator` but wrapped with a tqdm progress bar.

    Parameters
    ----------
    total : int
        Total number of elements.
    batch_size : int
        Maximum number of elements per batch.
    **tqdm_kwargs
        Additional keyword arguments forwarded to ``tqdm``.

    Returns
    -------
    tqdm
        A tqdm-wrapped iterator yielding ``(start, end)`` index pairs.
    """
    iterator = batch_iterator(total, batch_size)
    return tqdm(iterator, total=ceildiv(total, batch_size), **tqdm_kwargs)


def dict_product(input_dict):
    """Generate all possible dictionaries from a dict mapping keys to iterables.

    Computes the Cartesian product of the values and returns one dictionary
    per combination, each preserving the original keys.

    Parameters
    ----------
    input_dict : dict
        Dictionary mapping keys to iterables of possible values.

    Returns
    -------
    list of dict
        All combinations, each as a dictionary with the same keys.
    """
    # Extract keys and corresponding iterables
    keys, values = zip(*input_dict.items())

    # Generate all combinations using product
    combinations = itertools.product(*values)

    # Create a list of dictionaries for each combination
    result = [dict(zip(keys, combo)) for combo in combinations]

    return result
