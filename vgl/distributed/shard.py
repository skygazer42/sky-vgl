from dataclasses import dataclass, field
from pathlib import Path

import torch

from vgl.distributed.partition import PartitionManifest, PartitionShard, load_partition_manifest
from vgl.distributed.store import (
    _LazyPartitionFeatureStoreAdapter,
    _LazyPartitionGraphStoreAdapter,
    _PartitionStoreBundleCache,
    _partition_edge_types,
)
from vgl.graph.graph import Graph
from vgl.graph.schema import GraphSchema


EdgeType = tuple[str, str, str]


def _sorted_lookup_state(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
    if ids.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=ids.device)
        return empty, empty
    sorted_ids, sort_perm = torch.sort(ids, stable=True)
    return sorted_ids, sort_perm


def _lookup_local_positions(
    sorted_ids: torch.Tensor,
    sort_perm: torch.Tensor,
    ids: torch.Tensor,
    *,
    missing_message: str,
) -> torch.Tensor:
    ids = torch.as_tensor(ids, dtype=torch.long, device=sorted_ids.device).view(-1)
    if ids.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=sorted_ids.device)

    positions = torch.searchsorted(sorted_ids, ids)
    if bool((positions >= sorted_ids.numel()).any()):
        missing_id = int(ids[positions >= sorted_ids.numel()][0].item())
        raise KeyError(missing_message.format(id=missing_id))
    matched_ids = sorted_ids[positions]
    if bool((matched_ids != ids).any()):
        missing_id = int(ids[matched_ids != ids][0].item())
        raise KeyError(missing_message.format(id=missing_id))
    return sort_perm[positions]


def _manifest_node_ids_by_type(partition: PartitionShard) -> dict[str, torch.Tensor]:
    return {
        str(node_type): torch.arange(int(start), int(end), dtype=torch.long)
        for node_type, (start, end) in partition.node_ranges.items()
    }


def _ordered_edge_types(manifest: PartitionManifest, partition: PartitionShard) -> tuple[EdgeType, ...]:
    ordered = []
    seen = set()
    partition_edge_types = set(_partition_edge_types(partition))
    for edge_type in manifest.edge_types:
        edge_type = tuple(edge_type)
        if edge_type not in partition_edge_types or edge_type in seen:
            continue
        seen.add(edge_type)
        ordered.append(edge_type)
    if ordered:
        return tuple(ordered)
    return _partition_edge_types(partition)


def _schema_from_partition(manifest: PartitionManifest, partition: PartitionShard) -> GraphSchema:
    edge_types = _ordered_edge_types(manifest, partition)
    node_feature_shapes = partition.node_feature_shapes
    edge_feature_shapes = partition.edge_feature_shapes
    return GraphSchema(
        node_types=tuple(partition.node_ranges),
        edge_types=edge_types,
        node_features={
            node_type: tuple(node_feature_shapes.get(node_type, {}))
            for node_type in partition.node_ranges
        },
        edge_features={
            edge_type: ("edge_index",) + tuple(edge_feature_shapes.get(edge_type, {}))
            for edge_type in edge_types
        },
        time_attr=manifest.time_attr,
    )


@dataclass(slots=True)
class LocalGraphShard:
    manifest: PartitionManifest
    partition: PartitionShard
    root: Path
    node_ids_by_type: dict[str, torch.Tensor]
    feature_store: _LazyPartitionFeatureStoreAdapter
    graph_store: _LazyPartitionGraphStoreAdapter
    graph: Graph
    _cache: _PartitionStoreBundleCache = field(repr=False)
    _edge_types: tuple[EdgeType, ...] = field(default_factory=tuple, repr=False)
    _global_to_local_by_type: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict, repr=False)
    _global_to_local_edge_by_type: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._global_to_local_by_type = {
            node_type: _sorted_lookup_state(node_ids)
            for node_type, node_ids in self.node_ids_by_type.items()
        }
        edge_ids_by_type = self.partition.edge_ids_by_type
        self._global_to_local_edge_by_type = {
            edge_type: _sorted_lookup_state(torch.as_tensor(edge_ids_by_type[edge_type], dtype=torch.long))
            for edge_type in self._edge_types
            if edge_type in edge_ids_by_type
        }

    @property
    def boundary_edge_data_by_type(self) -> dict[EdgeType, dict]:
        return self._cache.bundle(self.partition.partition_id).boundary_edge_data_by_type

    def _resolve_edge_type(self, edge_type: EdgeType | None) -> EdgeType:
        if edge_type is not None:
            return tuple(edge_type)
        default_edge_type = ("node", "to", "node")
        if default_edge_type in self._edge_types:
            return default_edge_type
        if len(self._edge_types) == 1:
            return self._edge_types[0]
        raise KeyError("edge_type is required when multiple edge types exist")

    @property
    def node_ids(self) -> torch.Tensor:
        if set(self.node_ids_by_type) == {"node"}:
            return self.node_ids_by_type["node"]
        if "node" in self.node_ids_by_type:
            return self.node_ids_by_type["node"]
        raise AttributeError("node_ids is ambiguous for multi-type shards; use node_ids_for(node_type)")

    def node_ids_for(self, node_type: str = "node") -> torch.Tensor:
        try:
            return self.node_ids_by_type[str(node_type)]
        except KeyError as exc:
            raise KeyError(node_type) from exc

    @classmethod
    def from_partition_dir(cls, root, *, partition_id: int) -> "LocalGraphShard":
        root = Path(root)
        manifest = load_partition_manifest(root / "manifest.json")
        partition = next(
            (current for current in manifest.partitions if current.partition_id == int(partition_id)),
            None,
        )
        if partition is None:
            raise KeyError(f"unknown partition_id: {partition_id}")
        if partition.path is None:
            raise ValueError(f"partition {partition_id} does not declare a payload path")

        cache = _PartitionStoreBundleCache(root, {partition.partition_id: partition})
        feature_store = _LazyPartitionFeatureStoreAdapter(cache, partition)
        graph_store = _LazyPartitionGraphStoreAdapter(cache, partition)
        graph = Graph.from_storage(
            schema=_schema_from_partition(manifest, partition),
            feature_store=feature_store,
            graph_store=graph_store,
        )
        return cls(
            manifest=manifest,
            partition=partition,
            root=root,
            node_ids_by_type=_manifest_node_ids_by_type(partition),
            feature_store=feature_store,
            graph_store=graph_store,
            graph=graph,
            _cache=cache,
            _edge_types=_ordered_edge_types(manifest, partition),
        )

    def global_to_local(self, node_ids: torch.Tensor, *, node_type: str = "node") -> torch.Tensor:
        node_type = str(node_type)
        try:
            sorted_ids, sort_perm = self._global_to_local_by_type[node_type]
        except KeyError as exc:
            raise KeyError(node_type) from exc
        return _lookup_local_positions(
            sorted_ids,
            sort_perm,
            node_ids,
            missing_message=(
                f"node {{id}} is not present in partition {self.partition.partition_id} "
                f"for node type {node_type!r}"
            ),
        )

    def local_to_global(self, node_ids: torch.Tensor, *, node_type: str = "node") -> torch.Tensor:
        global_ids = self.node_ids_for(node_type)
        local_ids = torch.as_tensor(node_ids, dtype=torch.long)
        if local_ids.numel() > 0 and ((local_ids < 0).any() or (local_ids >= global_ids.numel()).any()):
            raise IndexError(
                f"local node ids are out of range for partition {self.partition.partition_id} and node type {node_type!r}"
            )
        return global_ids[local_ids]

    def edge_ids(self, *, edge_type=None) -> torch.Tensor:
        resolved = self._resolve_edge_type(edge_type)
        edge_ids_by_type = self.partition.edge_ids_by_type
        if resolved in edge_ids_by_type:
            return torch.as_tensor(edge_ids_by_type[resolved], dtype=torch.long)
        return torch.as_tensor(self.graph.edges[resolved].data["e_id"], dtype=torch.long)

    def boundary_edge_ids(self, *, edge_type=None) -> torch.Tensor:
        resolved = self._resolve_edge_type(edge_type)
        boundary_edge_ids_by_type = self.partition.boundary_edge_ids_by_type
        if resolved in boundary_edge_ids_by_type:
            return torch.as_tensor(boundary_edge_ids_by_type[resolved], dtype=torch.long)
        return torch.as_tensor(self.boundary_edge_data_by_type[resolved]["e_id"], dtype=torch.long)

    def incident_edge_ids(self, *, edge_type=None) -> torch.Tensor:
        return torch.cat((self.edge_ids(edge_type=edge_type), self.boundary_edge_ids(edge_type=edge_type)))

    def global_to_local_edge(self, edge_ids: torch.Tensor, *, edge_type=None) -> torch.Tensor:
        resolved = self._resolve_edge_type(edge_type)
        lookup_state = self._global_to_local_edge_by_type.get(resolved)
        if lookup_state is None:
            lookup_state = _sorted_lookup_state(self.edge_ids(edge_type=resolved))
            self._global_to_local_edge_by_type[resolved] = lookup_state
        sorted_ids, sort_perm = lookup_state
        return _lookup_local_positions(
            sorted_ids,
            sort_perm,
            edge_ids,
            missing_message=(
                f"edge {{id}} is not present in partition {self.partition.partition_id} "
                f"for edge type {resolved!r}"
            ),
        )

    def local_to_global_edge(self, edge_ids: torch.Tensor, *, edge_type=None) -> torch.Tensor:
        resolved = self._resolve_edge_type(edge_type)
        global_ids = self.edge_ids(edge_type=resolved)
        local_ids = torch.as_tensor(edge_ids, dtype=torch.long)
        if local_ids.numel() > 0 and ((local_ids < 0).any() or (local_ids >= global_ids.numel()).any()):
            raise IndexError(
                f"local edge ids are out of range for partition {self.partition.partition_id} and edge type {resolved!r}"
            )
        return global_ids[local_ids]

    def global_edge_index(self, *, edge_type=None) -> torch.Tensor:
        resolved = self._resolve_edge_type(edge_type)
        src_type, _, dst_type = resolved
        local_edge_index = self.graph_store.edge_index(resolved)
        return torch.stack(
            (
                self.local_to_global(local_edge_index[0], node_type=src_type),
                self.local_to_global(local_edge_index[1], node_type=dst_type),
            )
        )

    def boundary_edge_index(self, *, edge_type=None) -> torch.Tensor:
        resolved = self._resolve_edge_type(edge_type)
        return torch.as_tensor(self.boundary_edge_data_by_type[resolved]["edge_index"], dtype=torch.long)

    def incident_edge_index(self, *, edge_type=None) -> torch.Tensor:
        return torch.cat((self.global_edge_index(edge_type=edge_type), self.boundary_edge_index(edge_type=edge_type)), dim=1)


__all__ = ["LocalGraphShard"]
