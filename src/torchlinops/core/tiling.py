import torch


class VerticalSplit(LinearOperator):
    ...
    # Adjoint is Sum
    # Closer is VerticalStack


class VerticalStack(LinearOperator):
    ...
    # Adjoint is HorizontalSplit
    # Opener is VerticalSplit


class HoriontalSplit(LinearOperator):
    ...
    # Adjoint is VerticalStack
    # Closer is Sum

class Sum(LinearOperator):
    ...
    # Adjoint is VerticalSplit
    # Opener is HorizontalSplit
