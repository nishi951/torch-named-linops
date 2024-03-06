import copy


class CollectResultsCallback:
    def __init__(self):
        self.preds = []

    def __call__(self, s):
        self.preds.append(copy.deepcopy(s.pred))
        return s
