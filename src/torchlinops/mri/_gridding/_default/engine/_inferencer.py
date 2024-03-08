__all__ = ['AbstractInferencer']


class AbstractInferencer:
    """Base class for inferencer with event-driven handlers"""

    def val(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()
