from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from vgl.metrics.base import MetricSpec


@runtime_checkable
class TaskProtocol(Protocol):
    metrics: Sequence[MetricSpec] | None

    def loss(self, graph: Any, predictions: Any, stage: str):
        ...

    def targets(self, batch: Any, stage: str):
        ...

    def predictions_for_metrics(self, batch: Any, predictions: Any, stage: str):
        ...


class Task(TaskProtocol):
    metrics = None

    def loss(self, graph, predictions, stage):
        raise NotImplementedError

    def targets(self, batch, stage):
        raise NotImplementedError

    def predictions_for_metrics(self, batch, predictions, stage):
        del batch, stage
        return predictions
