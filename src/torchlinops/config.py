import warnings

import torchlinops

# Global config variables
# If True, completely eliminate any inner Identity linops inside a normal(inner) call
reduce_identity_in_normal = True

# If True, cache .H and .N results. Deprecated - caching adds complexity and
# can cause stale state issues. Will be removed in a future version.
cache_adjoint_normal = False


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
