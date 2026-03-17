class Evaluator:
    def __init__(self, metrics=None):
        self.metrics = metrics or []

    def evaluate(self, predictions, targets):
        return {
            metric.__class__.__name__: metric(predictions, targets)
            for metric in self.metrics
        }
