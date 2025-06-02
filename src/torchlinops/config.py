import torchlinops

# Global config variables
reduce_identity_in_normal = True


def inner_not_relevant(inner):
    return (inner is None) or (
        isinstance(inner, torchlinops.Identity) and reduce_identity_in_normal
    )
