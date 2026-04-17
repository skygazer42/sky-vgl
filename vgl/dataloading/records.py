from dataclasses import dataclass, field
from typing import Any

_COMMON_OPTIONAL_FIELD_NAMES = ("metadata", "sample_id", "query_id")


def _resolved_metadata_value(explicit_value, metadata: dict[str, Any], *, key: str):
    if explicit_value is not None:
        return explicit_value
    return metadata.get(key)


class _SampleRecordContract:
    optional_field_names = _COMMON_OPTIONAL_FIELD_NAMES
    seed_field_names: tuple[str, ...] = ()

    @property
    def seed_fields(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.seed_field_names}


@dataclass(slots=True)
class SampleRecord(_SampleRecordContract):
    graph: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    query_id: Any | None = None
    source_graph_id: str | None = None
    subgraph_seed: Any | None = None
    blocks: list[Any] | None = None
    seed_field_names = ("subgraph_seed",)

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
class LinkPredictionRecord(_SampleRecordContract):
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
    seed_field_names = ("src_index", "dst_index")

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
class TemporalEventRecord(_SampleRecordContract):
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
    seed_field_names = ("src_index", "dst_index", "timestamp")

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
