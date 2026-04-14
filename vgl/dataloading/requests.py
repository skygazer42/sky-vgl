from dataclasses import dataclass, field
from typing import Any

import torch


def _resolved_metadata_value(explicit_value, metadata: dict[str, Any], *, key: str):
    if explicit_value is not None:
        return explicit_value
    return metadata.get(key)


def _as_rank1_tensor(value, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    _require_rank1(tensor, name=name)
    return tensor


def _require_rank1(value: torch.Tensor, *, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank-1")


def _require_same_length(*values: torch.Tensor) -> None:
    lengths = {int(value.numel()) for value in values}
    if len(lengths) > 1:
        raise ValueError("seed tensors must have the same length")


@dataclass(slots=True)
class NodeSeedRequest:
    node_ids: torch.Tensor
    node_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    query_id: Any | None = None

    def __post_init__(self) -> None:
        self.node_ids = _as_rank1_tensor(self.node_ids, name="node_ids")

    @property
    def kind(self) -> str:
        return "node"

    @property
    def seed_count(self) -> int:
        return int(self.node_ids.numel())

    @property
    def seed_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.node_ids,)

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
class LinkSeedRequest:
    src_ids: torch.Tensor
    dst_ids: torch.Tensor
    edge_type: Any = None
    labels: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    query_id: Any | None = None

    def __post_init__(self) -> None:
        self.src_ids = _as_rank1_tensor(self.src_ids, name="src_ids")
        self.dst_ids = _as_rank1_tensor(self.dst_ids, name="dst_ids")
        values = [self.src_ids, self.dst_ids]
        if self.labels is not None:
            self.labels = _as_rank1_tensor(self.labels, name="labels")
            values.append(self.labels)
        _require_same_length(*values)

    @property
    def kind(self) -> str:
        return "link"

    @property
    def seed_count(self) -> int:
        return int(self.src_ids.numel())

    @property
    def seed_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.src_ids, self.dst_ids)

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
class TemporalSeedRequest:
    src_ids: torch.Tensor
    dst_ids: torch.Tensor
    timestamps: torch.Tensor
    edge_type: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    query_id: Any | None = None

    def __post_init__(self) -> None:
        self.src_ids = _as_rank1_tensor(self.src_ids, name="src_ids")
        self.dst_ids = _as_rank1_tensor(self.dst_ids, name="dst_ids")
        self.timestamps = _as_rank1_tensor(self.timestamps, name="timestamps")
        _require_same_length(self.src_ids, self.dst_ids, self.timestamps)

    @property
    def kind(self) -> str:
        return "temporal"

    @property
    def seed_count(self) -> int:
        return int(self.src_ids.numel())

    @property
    def seed_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.src_ids, self.dst_ids, self.timestamps)

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
class GraphSeedRequest:
    graph_ids: torch.Tensor
    labels: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    query_id: Any | None = None

    def __post_init__(self) -> None:
        self.graph_ids = _as_rank1_tensor(self.graph_ids, name="graph_ids")
        if self.labels is not None:
            self.labels = _as_rank1_tensor(self.labels, name="labels")
            _require_same_length(self.graph_ids, self.labels)

    @property
    def kind(self) -> str:
        return "graph"

    @property
    def seed_count(self) -> int:
        return int(self.graph_ids.numel())

    @property
    def seed_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.graph_ids,)

    @property
    def resolved_sample_id(self):
        return _resolved_metadata_value(self.sample_id, self.metadata, key="sample_id")

    @property
    def resolved_query_id(self):
        query_id = _resolved_metadata_value(self.query_id, self.metadata, key="query_id")
        if query_id is not None:
            return query_id
        return self.resolved_sample_id
