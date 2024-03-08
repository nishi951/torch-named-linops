import itertools

__all__ = [
    'batch_iterator',
    'dict_product',
]

def batch_iterator(total, batch_size):
    assert total > 0, f'batch_iterator called with {total} elements'
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])

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
