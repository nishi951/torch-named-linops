
@dataclass
class InferenceState:
    model: nn.Module
    features: Optional[Mapping] = None
    pred: Optional[Mapping] = None
    target: Optional[Mapping] = None
    global_step: int = -1
    epoch: int = -1

class BasicInferencer:
    """Apply a trained model to held-out data

    Validation and Testing are combined under "Inference" in that in both
    cases, the model itself should not glean any information from the data applied.
    However, future models may be trained differently depending on the results of validation

    Testing should only be performed once, typically at the very end.
    """
    def __init__(self, hparams: BasicInferencerHparams, handlers=None):
        self.hparams = hparams
        self.handlers = defaultdict(list)

        # Default handlers
        if handlers is None:
            self.register_handler('val_step_ended', GlobalStep())

    def initialize_inference_state(model):
        model.eval()
        s = InferenceState(model)
        return s

    def val(self, model: nn.Module, val_dataloader):
        """Run a validation

        Validation occurs during the model iteration process. While it is like
        testing in that the model is applied to unseen data, it is
        distinct from testing in that the model's hyperparameters can be updated
        based on results from validation.
        """
        s = self.initialize_inference_state(model)
        self.dispatch('val_started', s)
        preds = []
        for s.features, s.target in tqdm(iter(val_dataloader), 'Validation'):
            self.dispatch('val_step_started', s)
            s.pred = self.infer(s.model, s.features)
            preds.append(s.pred) # TODO: move to cpu ?
            self.dispatch('val_step_ended', s)
        self.dispatch('val_ended', s)
        return preds

    def test(self, model: nn.Module, test_dataloader):
        """Run a test
        """
        s = self.initialize_inference_state(model)
        self.dispatch('test_started', s)
        preds = []
        for features, _ in tqdm(iter(test_dataloader), 'Testing'):
            self.dispatch('test_step_started', s)
            pred = self.infer(model, features)
            preds.append(pred)
            self.dispatch('test_step_ended', s)
        self.dispatch('test_ended', s)
        return preds

    @torch.no_grad()
    def infer(self, model, features):
        """Convenience function
        """
        return model(features)
