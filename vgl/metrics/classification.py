from copy import deepcopy

import torch

from vgl.metrics.base import Metric
from vgl.metrics.base import MetricProtocol, MetricSpec


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


class Accuracy(Metric):
    name = "accuracy"

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions, targets, *, batch=None):
        del batch
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
        self.correct += _as_python_int((predicted == targets).sum())
        self.total += int(targets.numel())

    def compute(self):
        if self.total == 0:
            raise ValueError("Accuracy requires at least one example before compute()")
        return self.correct / self.total


def build_metric(metric: MetricSpec) -> MetricProtocol:
    if isinstance(metric, MetricProtocol):
        return deepcopy(metric)
    if metric == "accuracy":
        return Accuracy()
    if metric == "mrr":
        from vgl.metrics.ranking import MRR

        return MRR()
    if metric == "filtered_mrr":
        from vgl.metrics.ranking import FilteredMRR

        return FilteredMRR()
    if isinstance(metric, str) and metric.startswith("hits@"):
        from vgl.metrics.ranking import HitsAtK

        suffix = metric.split("@", 1)[1]
        if not suffix.isdigit():
            raise ValueError(f"Unsupported metric: {metric}")
        return HitsAtK(int(suffix))
    if isinstance(metric, str) and metric.startswith("filtered_hits@"):
        from vgl.metrics.ranking import FilteredHitsAtK

        suffix = metric.split("@", 1)[1]
        if not suffix.isdigit():
            raise ValueError(f"Unsupported metric: {metric}")
        return FilteredHitsAtK(int(suffix))
    raise ValueError(f"Unsupported metric: {metric}")
