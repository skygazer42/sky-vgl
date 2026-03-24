from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PlanStage:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    request: Any
    stages: tuple[PlanStage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    graph: Any | None = None

    def append(self, *stages: PlanStage) -> "SamplingPlan":
        return SamplingPlan(
            request=self.request,
            stages=self.stages + tuple(stages),
            metadata=dict(self.metadata),
            graph=self.graph,
        )
