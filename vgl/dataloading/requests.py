from dataclasses import dataclass, field
from typing import Any

import torch


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

    def __post_init__(self) -> None:
        _require_rank1(self.node_ids, name="node_ids")

    @property
    def kind(self) -> str:
        return "node"


@dataclass(slots=True)
class LinkSeedRequest:
    src_ids: torch.Tensor
    dst_ids: torch.Tensor
    edge_type: Any = None
    labels: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_rank1(self.src_ids, name="src_ids")
        _require_rank1(self.dst_ids, name="dst_ids")
        values = [self.src_ids, self.dst_ids]
        if self.labels is not None:
            _require_rank1(self.labels, name="labels")
            values.append(self.labels)
        _require_same_length(*values)

    @property
    def kind(self) -> str:
        return "link"


@dataclass(slots=True)
class TemporalSeedRequest:
    src_ids: torch.Tensor
    dst_ids: torch.Tensor
    timestamps: torch.Tensor
    edge_type: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_rank1(self.src_ids, name="src_ids")
        _require_rank1(self.dst_ids, name="dst_ids")
        _require_rank1(self.timestamps, name="timestamps")
        _require_same_length(self.src_ids, self.dst_ids, self.timestamps)

    @property
    def kind(self) -> str:
        return "temporal"


@dataclass(slots=True)
class GraphSeedRequest:
    graph_ids: torch.Tensor
    labels: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_rank1(self.graph_ids, name="graph_ids")
        if self.labels is not None:
            _require_rank1(self.labels, name="labels")
            _require_same_length(self.graph_ids, self.labels)

    @property
    def kind(self) -> str:
        return "graph"
