"""Global configuration options for torchlinops.

This module provides global variables that control the behavior of the library.
Import and modify these in your code to customize torchlinops behavior.

Example
-------
>>> import torchlinops.config as config
>>> config.log_device_transfers = True  # Enable debug logging
>>> with config.using(log_device_transfers=False):
...     pass  # Temporarily disable logging

Variables
---------
reduce_identity_in_normal : bool
    If True, completely eliminate any inner Identity linops inside a normal(inner)
    call. Default is True.
cache_adjoint_normal : bool
    If True, cache .H and .N results. Deprecated - caching adds complexity and can
    cause stale state issues. Will be removed in a future version. Default is False.
log_device_transfers : bool
    If True, log CUDA events creation, stream synchronization, and device transfers
    in the ToDevice linop and related utilities. Default is True.
"""

import warnings

import torchlinops

# Global config variables
# If True, completely eliminate any inner Identity linops inside a normal(inner) call
reduce_identity_in_normal = True

# If True, cache .H and .N results. Deprecated - caching adds complexity and
# can cause stale state issues. Will be removed in a future version.
cache_adjoint_normal = False

# If True, log CUDA events creation, stream synchronization, and device transfers
# in the ToDevice linop and related utilities.
log_device_transfers = True


def inner_not_relevant(inner):
    return (inner is None) or (
        isinstance(inner, torchlinops.Identity) and reduce_identity_in_normal
    )


def _warn_if_caching_enabled():
    if cache_adjoint_normal:
        warnings.warn(
            "cache_adjoint_normal is deprecated and will be removed in a future version. "
            "Caching adjoint/normal operators can lead to stale state bugs. "
            "Please set torchlinops.config.cache_adjoint_normal = False.",
            FutureWarning,
            stacklevel=4,
        )


class ConfigContext:
    """Context manager for temporarily modifying config values.

    Example
    -------
    >>> import torchlinops.config as config
    >>> with config.using(reduce_identity_in_normal=False):
    ...     # reduce_identity_in_normal is False here
    ...     pass
    >>> # original value restored
    """

    VALID_KEYS = {
        "reduce_identity_in_normal",
        "cache_adjoint_normal",
        "log_device_transfers",
    }

    def __init__(self, **kwargs: bool) -> None:
        for key in kwargs:
            if key not in self.VALID_KEYS:
                raise ValueError(
                    f"Invalid config key: {key!r}. Valid keys: {self.VALID_KEYS}"
                )
        self._changes = kwargs

    def __enter__(self) -> "ConfigContext":
        self._saved: dict[str, bool] = {}
        for key, value in self._changes.items():
            self._saved[key] = getattr(torchlinops.config, key)
            setattr(torchlinops.config, key, value)
        return self

    def __exit__(self, *args: object) -> None:
        for key, value in self._saved.items():
            setattr(torchlinops.config, key, value)


def using(**kwargs: bool) -> ConfigContext:
    """Create a context manager for temporarily modifying config values.

    Example
    -------
    >>> import torchlinops.config as config
    >>> with config.using(reduce_identity_in_normal=False):
    ...     # reduce_identity_in_normal is False here
    ...     pass
    >>> # original value restored

    Parameters
    ----------
    **kwargs : bool
        Config values to temporarily set. Valid keys are:
        - reduce_identity_in_normal
        - cache_adjoint_normal
        - log_device_transfers

    Returns
    -------
    ConfigContext
        A context manager that restores original values on exit.

    Raises
    ------
    ValueError
        If any key is not a valid config variable name.
    """
    return ConfigContext(**kwargs)
