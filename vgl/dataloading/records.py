from dataclasses import dataclass, field
from typing import Any


def _resolved_metadata_value(explicit_value, metadata: dict[str, Any], *, key: str):
    if explicit_value is not None:
        return explicit_value
    return metadata.get(key)


@dataclass(slots=True)
class SampleRecord:
    graph: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    query_id: Any | None = None
    source_graph_id: str | None = None
    subgraph_seed: Any | None = None
    blocks: list[Any] | None = None

    @property
    def kind(self) -> str:
        return "node"

    @property
    def seed_count(self) -> int:
        return 1

    @property
    def resolved_sample_id(self):
        return _resolved_metadata_value(self.sample_id, self.metadata, key="sample_id")

    @property
    def resolved_query_id(self):
        query_id = _resolved_metadata_value(self.query_id, self.metadata, key="query_id")
        if query_id is not None:
            return query_id
        return self.resolved_sample_id


@dataclass(slots=True)
class LinkPredictionRecord:
    graph: Any
    src_index: int
    dst_index: int
    label: int
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    exclude_seed_edge: bool = False
    hard_negative_dst: Any | None = None
    candidate_dst: Any | None = None
    edge_type: Any | None = None
    reverse_edge_type: Any | None = None
    query_id: Any | None = None
    filter_ranking: bool = False
    blocks: list[Any] | None = None

    @property
    def kind(self) -> str:
        return "link"

    @property
    def seed_count(self) -> int:
        return 1

    @property
    def resolved_sample_id(self):
        return _resolved_metadata_value(self.sample_id, self.metadata, key="sample_id")

    @property
    def resolved_query_id(self):
        query_id = _resolved_metadata_value(self.query_id, self.metadata, key="query_id")
        if query_id is not None:
            return query_id
        return self.resolved_sample_id


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
    query_id: Any | None = None
    edge_type: Any | None = None

    @property
    def kind(self) -> str:
        return "temporal"

    @property
    def seed_count(self) -> int:
        return 1

    @property
    def resolved_sample_id(self):
        return _resolved_metadata_value(self.sample_id, self.metadata, key="sample_id")

    @property
    def resolved_query_id(self):
        query_id = _resolved_metadata_value(self.query_id, self.metadata, key="query_id")
        if query_id is not None:
            return query_id
        return self.resolved_sample_id
