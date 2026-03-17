from copy import deepcopy

from vgl.metrics.base import Metric


class Accuracy(Metric):
    name = "accuracy"

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions, targets):
        if predictions.ndim == targets.ndim + 1 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)
        if predictions.ndim == targets.ndim + 1:
            predicted = predictions.argmax(dim=-1)
        elif predictions.ndim == targets.ndim:
            predicted = (predictions >= 0).to(dtype=targets.dtype)
        else:
            raise ValueError("Accuracy received incompatible prediction/target shape")
        if predicted.shape != targets.shape:
            raise ValueError("Accuracy received incompatible prediction/target shape")
        self.correct += int((predicted == targets).sum().item())
        self.total += int(targets.numel())

    def compute(self):
        if self.total == 0:
            raise ValueError("Accuracy requires at least one example before compute()")
        return self.correct / self.total


def build_metric(metric):
    if isinstance(metric, Metric):
        return deepcopy(metric)
    if metric == "accuracy":
        return Accuracy()
    raise ValueError(f"Unsupported metric: {metric}")
