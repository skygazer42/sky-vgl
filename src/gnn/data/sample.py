from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SampleRecord:
    graph: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    source_graph_id: str | None = None
    subgraph_seed: Any | None = None


@dataclass(slots=True)
class TemporalEventRecord:
    graph: Any
    src_index: int
    dst_index: int
    timestamp: int
    label: int
    event_features: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
