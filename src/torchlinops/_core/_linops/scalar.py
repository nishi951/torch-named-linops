from .diagonal import Diagonal


class Scalar(Diagonal):
    """The result of scalar multiplication

    A Diagonal linop that is trivially splittable.
    """

    def split_forward_fn(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, "Scalar linop must be split identically"
        return weight

    def size_fn(self, dim: str, weight):
        return None
