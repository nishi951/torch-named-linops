import itertools

from tqdm import tqdm

__all__ = [
    "ceildiv",
    "batch_iterator",
    "batch_tqdm",
    "dict_product",
]


def ceildiv(dividend, divisor):
    return -(-dividend // divisor)


def batch_iterator(total, batch_size):
    assert total > 0, f"batch_iterator called with {total} elements"
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])


def batch_tqdm(total, batch_size, **tqdm_kwargs):
    iterator = batch_iterator(total, batch_size)
    return tqdm(iterator, total=ceildiv(total, batch_size), **tqdm_kwargs)


def dict_product(input_dict):
    """Generate all possible dictionaries from a dictionary
    mapping keys to iterables

    ChatGPT-4
    """
    # Extract keys and corresponding iterables
    keys, values = zip(*input_dict.items())

    # Generate all combinations using product
    combinations = itertools.product(*values)

    # Create a list of dictionaries for each combination
    result = [dict(zip(keys, combo)) for combo in combinations]

    return result
