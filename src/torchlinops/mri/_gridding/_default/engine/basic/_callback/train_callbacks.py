from utils.event import Callback, EventManager


class StepCallback:
    """Increment steps as training proceeds."""

    def __init__(self, fieldname):
        self.fieldname = fieldname

    def __call__(self, s):
        setattr(s, self.fieldname, getattr(s, self.fieldname) + 1)
        return s


class LogLossCallback:
    def __init__(self):
        self.losses = []

    def __call__(self, s):
        self.losses.append(s.loss.item())
        return s


class TrainManager(EventManager):
    """Default callbacks for training"""

    def __init__(self):
        self.register_handler(
            'train_step_ended',
            Callback('TrainStepCallback', StepCallback('global_step')),
        )
        self.register_handler(
            'epoch_ended', Callback('TrainStepCallback', StepCallback('epoch'))
        )
        self.log_loss_callback = LogLossCallback()
        self.register_handler(
            'train_step_ended', Callback('LogLossCallback', self.log_loss_callback)
        )

    @property
    def losses(self):
        """Easy access to losses"""
        return self.log_loss_callback.losses
