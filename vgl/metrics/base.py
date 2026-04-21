from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable


@runtime_checkable
class MetricProtocol(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def update(self, predictions: Any, targets: Any, **kwargs: Any) -> None:
        ...

    def compute(self) -> int | float:
        ...


MetricSpec: TypeAlias = MetricProtocol | str


class Metric(MetricProtocol):
    name = "metric"

    def reset(self):
        raise NotImplementedError

    def update(self, predictions, targets, **kwargs):
        del kwargs
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError
