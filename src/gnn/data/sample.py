from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SampleRecord:
    graph: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    source_graph_id: str | None = None
    subgraph_seed: Any | None = None
