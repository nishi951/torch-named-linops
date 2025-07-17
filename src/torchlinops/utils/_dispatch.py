import inspect

__all__ = ["check_signature"]


def check_signature(signature_spec: list, *args, **kwargs):
    """Check if the provided arguments match the specified signature.

    Parameters
    ----------
    signature_spec : list
        A list that defines the function signature.
    *args : tuple
        Positional arguments to check against the signature.
    **kwargs : dict
        Keyword arguments to check against the signature.

    Returns
    -------
    BoundArguments or None
        The bound signature if the arguments match the signature, None otherwise.

    Examples
    --------
    >>> check_signature([('arg1', int), ('arg2', str)], 10, 'test')
    True
    >>> check_signature([('arg1', int), ('arg2', str)], 'test', 10)
    False
    >>> check_signature([('arg1', int)], arg1=5)
    True
    >>> check_signature([('arg1', int)], arg1='not an int')
    False
    """
    sig, allow_kwargs = build_signature(signature_spec)
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError:
        return None

    for name, val in bound.arguments.items():
        expected = sig.parameters[name].annotation
        if expected is not inspect.Parameter.empty and not isinstance(val, expected):
            return None

    if not allow_kwargs and bound.kwargs:
        return None

    return bound


def build_signature(signature_spec: list):
    """Build a signature from a list of tokens.

    Parameters
    ----------
    signature_spec : list
        A list of tokens that define the function signature.
        Tokens can be strings or tuples with varying formats as detailed below.

    Returns
    -------
    inspect.Signature
        A Signature object representing the constructed function signature.
    """
    params = []
    seen_star = False

    for token in signature_spec:
        if token == "/":
            kind = inspect.Parameter.POSITIONAL_ONLY

        elif token == "*":
            seen_star = True
            continue  # next params will be KEYWORD_ONLY

        elif token == "**":
            # Anonymous **kwargs
            param = inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
            params.append(param)

        elif isinstance(token, str) and token.startswith("**"):
            name = token[2:]
            param = inspect.Parameter(name, inspect.Parameter.VAR_KEYWORD)
            params.append(param)

        elif isinstance(token, str) and token.startswith("*"):
            name = token[1:]
            seen_star = True
            param = inspect.Parameter(name, inspect.Parameter.VAR_POSITIONAL)
            params.append(param)

        elif isinstance(token, tuple):
            if not (1 <= len(token) <= 3):
                raise ValueError(f"Invalid tuple token: {token}")

            name = token[0]
            annotation = inspect.Parameter.empty
            default = inspect.Parameter.empty

            if len(token) == 2:
                # Annotation or default?
                if isinstance(token[1], type):
                    annotation = token[1]
                else:
                    default = token[1]
            elif len(token) == 3:
                annotation, default = token[1], token[2]

            kind = (
                inspect.Parameter.KEYWORD_ONLY
                if seen_star
                else inspect.Parameter.POSITIONAL_OR_KEYWORD
            )

            param = inspect.Parameter(
                name, kind, annotation=annotation, default=default
            )
            params.append(param)

        else:
            raise ValueError(f"Invalid token: {token}")

    return inspect.Signature(params)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
