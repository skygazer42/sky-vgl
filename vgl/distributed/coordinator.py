from dataclasses import dataclass
from typing import Protocol

import torch

from vgl.distributed.partition import PartitionManifest, PartitionShard
from vgl.distributed.shard import LocalGraphShard
from vgl.distributed.store import (
    DistributedFeatureStore,
    DistributedGraphStore,
    LocalFeatureStoreAdapter,
    load_partitioned_stores,
)
from vgl.sparse import SparseLayout, SparseTensor
from vgl.storage.base import TensorSlice
from vgl.storage.feature_store import FeatureKey


@dataclass(frozen=True, slots=True)
class ShardRoute:
    partition_id: int
    global_ids: torch.Tensor
    local_ids: torch.Tensor
    positions: torch.Tensor

    def __post_init__(self) -> None:
        if self.global_ids.ndim != 1 or self.local_ids.ndim != 1 or self.positions.ndim != 1:
            raise ValueError("ShardRoute tensors must be rank-1")
        if not (self.global_ids.numel() == self.local_ids.numel() == self.positions.numel()):
            raise ValueError("ShardRoute tensors must have the same length")


class SamplingCoordinator(Protocol):
    def partition_ids(self) -> tuple[int, ...]:
        ...

    def route_node_ids(self, node_ids: torch.Tensor, *, node_type: str = "node") -> tuple[ShardRoute, ...]:
        ...

    def route_edge_ids(self, edge_ids: torch.Tensor, *, edge_type=None) -> tuple[ShardRoute, ...]:
        ...

    def fetch_node_features(self, key: FeatureKey, node_ids: torch.Tensor) -> TensorSlice:
        ...

    def fetch_edge_features(self, key: FeatureKey, edge_ids: torch.Tensor) -> TensorSlice:
        ...

    def partition_node_ids(self, partition_id: int, *, node_type: str = "node") -> torch.Tensor:
        ...

    def partition_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        ...

    def partition_boundary_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        ...

    def partition_incident_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        ...

    def fetch_partition_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
        global_ids: bool = False,
    ) -> torch.Tensor:
        ...

    def fetch_partition_boundary_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
    ) -> torch.Tensor:
        ...

    def fetch_partition_incident_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
    ) -> torch.Tensor:
        ...

    def fetch_partition_adjacency(
        self,
        partition_id: int,
        *,
        edge_type=None,
        layout: SparseLayout | str = SparseLayout.COO,
    ) -> SparseTensor:
        ...


def _resolve_edge_type(edge_type, edge_types) -> tuple[str, str, str]:
    if edge_type is not None:
        return tuple(edge_type)
    edge_types = tuple(tuple(current) for current in edge_types)
    default_edge_type = ("node", "to", "node")
    if default_edge_type in edge_types:
        return default_edge_type
    if len(edge_types) == 1:
        return edge_types[0]
    raise KeyError("edge_type is required when multiple edge types exist")


def _build_routes(grouped: dict[int, dict[str, list[int]]]) -> tuple[ShardRoute, ...]:
    routes = []
    for partition_id in sorted(grouped):
        bucket = grouped[partition_id]
        routes.append(
            ShardRoute(
                partition_id=partition_id,
                global_ids=torch.tensor(bucket["global_ids"], dtype=torch.long),
                local_ids=torch.tensor(bucket["local_ids"], dtype=torch.long),
                positions=torch.tensor(bucket["positions"], dtype=torch.long),
            )
        )
    return tuple(routes)


class LocalSamplingCoordinator:
    def __init__(self, shards: dict[int, LocalGraphShard]):
        if not shards:
            raise ValueError("LocalSamplingCoordinator requires at least one shard")
        self.shards = dict(shards)
        first_shard = next(iter(self.shards.values()))
        self.manifest = first_shard.manifest
        self.feature_stores: dict[int, DistributedFeatureStore] = {
            partition_id: LocalFeatureStoreAdapter(shard.feature_store)
            for partition_id, shard in self.shards.items()
        }
        self._edge_partition_by_type: dict[object, dict[int, int]] = {}
        for partition_id, shard in self.shards.items():
            for edge_type in shard.graph.edges:
                owner = self._edge_partition_by_type.setdefault(edge_type, {})
                for edge_id in shard.edge_ids(edge_type=edge_type).tolist():
                    edge_id = int(edge_id)
                    if edge_id in owner:
                        raise ValueError(f"duplicate global edge id {edge_id} for edge type {edge_type!r}")
                    owner[edge_id] = partition_id

    def partition_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.shards))

    def _shard(self, partition_id: int) -> LocalGraphShard:
        try:
            return self.shards[int(partition_id)]
        except KeyError as exc:
            raise KeyError(f"missing shard for partition {partition_id}") from exc

    def partition_node_ids(self, partition_id: int, *, node_type: str = "node") -> torch.Tensor:
        return self._shard(partition_id).node_ids_for(node_type)

    def partition_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        return self._shard(partition_id).edge_ids(edge_type=edge_type)

    def partition_boundary_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        return self._shard(partition_id).boundary_edge_ids(edge_type=edge_type)

    def partition_incident_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        return self._shard(partition_id).incident_edge_ids(edge_type=edge_type)

    def fetch_partition_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
        global_ids: bool = False,
    ) -> torch.Tensor:
        shard = self._shard(partition_id)
        if global_ids:
            return shard.global_edge_index(edge_type=edge_type)
        return shard.graph_store.edge_index(edge_type)

    def fetch_partition_boundary_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
    ) -> torch.Tensor:
        return self._shard(partition_id).boundary_edge_index(edge_type=edge_type)

    def fetch_partition_incident_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
    ) -> torch.Tensor:
        return self._shard(partition_id).incident_edge_index(edge_type=edge_type)

    def fetch_partition_adjacency(
        self,
        partition_id: int,
        *,
        edge_type=None,
        layout: SparseLayout | str = SparseLayout.COO,
    ) -> SparseTensor:
        return self._shard(partition_id).graph_store.adjacency(edge_type=edge_type, layout=layout)

    def route_node_ids(self, node_ids: torch.Tensor, *, node_type: str = "node") -> tuple[ShardRoute, ...]:
        node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        grouped: dict[int, dict[str, list[int]]] = {}
        for position, node_id in enumerate(node_ids.tolist()):
            partition = self.manifest.owner(node_id, node_type=node_type)
            shard = self.shards.get(partition.partition_id)
            if shard is None:
                raise KeyError(f"missing shard for partition {partition.partition_id}")
            bucket = grouped.setdefault(
                partition.partition_id,
                {"global_ids": [], "local_ids": [], "positions": []},
            )
            bucket["global_ids"].append(int(node_id))
            bucket["positions"].append(int(position))
            bucket["local_ids"].append(int(shard.global_to_local(torch.tensor([node_id]), node_type=node_type).item()))
        return _build_routes(grouped)

    def route_edge_ids(self, edge_ids: torch.Tensor, *, edge_type=None) -> tuple[ShardRoute, ...]:
        edge_type = _resolve_edge_type(edge_type, next(iter(self.shards.values())).graph.edges)
        try:
            owners = self._edge_partition_by_type[edge_type]
        except KeyError as exc:
            raise KeyError(edge_type) from exc

        edge_ids = torch.as_tensor(edge_ids, dtype=torch.long)
        grouped: dict[int, dict[str, list[int]]] = {}
        for position, edge_id in enumerate(edge_ids.tolist()):
            try:
                partition_id = owners[int(edge_id)]
            except KeyError as exc:
                raise KeyError(f"edge {edge_id} is not present for edge type {edge_type!r}") from exc
            shard = self.shards.get(partition_id)
            if shard is None:
                raise KeyError(f"missing shard for partition {partition_id}")
            bucket = grouped.setdefault(
                partition_id,
                {"global_ids": [], "local_ids": [], "positions": []},
            )
            bucket["global_ids"].append(int(edge_id))
            bucket["positions"].append(int(position))
            bucket["local_ids"].append(int(shard.global_to_local_edge(torch.tensor([edge_id]), edge_type=edge_type).item()))
        return _build_routes(grouped)

    def fetch_node_features(self, key: FeatureKey, node_ids: torch.Tensor) -> TensorSlice:
        entity_kind, type_key, _ = key
        if entity_kind != "node":
            raise ValueError("fetch_node_features requires a node feature key")
        node_type = str(type_key)
        node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        routes = self.route_node_ids(node_ids, node_type=node_type)
        if node_ids.numel() == 0:
            first_store = next(iter(self.feature_stores.values()))
            shape = first_store.shape(key)
            return TensorSlice(index=node_ids, values=torch.empty((0,) + shape[1:]))

        values = None
        for route in routes:
            fetched = self.feature_stores[route.partition_id].fetch(
                key,
                route.local_ids,
                partition_id=route.partition_id,
            )
            if values is None:
                values = torch.empty(
                    (node_ids.numel(),) + tuple(fetched.values.shape[1:]),
                    dtype=fetched.values.dtype,
                    device=fetched.values.device,
                )
            values[route.positions] = fetched.values
        return TensorSlice(index=node_ids, values=values)

    def fetch_edge_features(self, key: FeatureKey, edge_ids: torch.Tensor) -> TensorSlice:
        entity_kind, type_key, _ = key
        if entity_kind != "edge":
            raise ValueError("fetch_edge_features requires an edge feature key")
        edge_type = tuple(type_key)
        edge_ids = torch.as_tensor(edge_ids, dtype=torch.long)
        routes = self.route_edge_ids(edge_ids, edge_type=edge_type)
        if edge_ids.numel() == 0:
            first_store = next(iter(self.feature_stores.values()))
            shape = first_store.shape(key)
            return TensorSlice(index=edge_ids, values=torch.empty((0,) + shape[1:]))

        values = None
        for route in routes:
            fetched = self.feature_stores[route.partition_id].fetch(
                key,
                route.local_ids,
                partition_id=route.partition_id,
            )
            if values is None:
                values = torch.empty(
                    (edge_ids.numel(),) + tuple(fetched.values.shape[1:]),
                    dtype=fetched.values.dtype,
                    device=fetched.values.device,
                )
            values[route.positions] = fetched.values
        return TensorSlice(index=edge_ids, values=values)


class StoreBackedSamplingCoordinator:
    def __init__(
        self,
        *,
        manifest: PartitionManifest,
        feature_store: DistributedFeatureStore,
        graph_store: DistributedGraphStore,
    ):
        self.manifest = manifest
        self.feature_store = feature_store
        self.graph_store = graph_store
        self._partitions = {partition.partition_id: partition for partition in manifest.partitions}
        self._edge_types = self._collect_edge_types()
        self._edge_ids_by_partition_and_type: dict[int, dict[tuple[str, str, str], tuple[int, ...]]] = {}
        self._boundary_edge_ids_by_partition_and_type: dict[int, dict[tuple[str, str, str], tuple[int, ...]]] = {}
        self._edge_partition_by_type: dict[tuple[str, str, str], dict[int, int]] = {}
        self._boundary_partition_by_type: dict[tuple[str, str, str], dict[int, int]] = {}
        self._local_edge_id_by_partition_and_type: dict[int, dict[tuple[str, str, str], dict[int, int]]] = {}

        for partition in manifest.partitions:
            edge_ids_by_type = {}
            boundary_edge_ids_by_type = {}
            local_edge_ids_by_type = {}
            for edge_type in self._edge_types:
                edge_ids = tuple(int(edge_id) for edge_id in partition.edge_ids_by_type.get(edge_type, ()))
                boundary_edge_ids = tuple(int(edge_id) for edge_id in partition.boundary_edge_ids_by_type.get(edge_type, ()))
                edge_ids_by_type[edge_type] = edge_ids
                boundary_edge_ids_by_type[edge_type] = boundary_edge_ids
                local_edge_ids_by_type[edge_type] = {edge_id: index for index, edge_id in enumerate(edge_ids)}
                owners = self._edge_partition_by_type.setdefault(edge_type, {})
                boundary_owners = self._boundary_partition_by_type.setdefault(edge_type, {})
                for edge_id in edge_ids:
                    if edge_id in owners:
                        raise ValueError(f"duplicate global edge id {edge_id} for edge type {edge_type!r}")
                    owners[edge_id] = partition.partition_id
                for edge_id in boundary_edge_ids:
                    boundary_owners.setdefault(edge_id, partition.partition_id)
            self._edge_ids_by_partition_and_type[partition.partition_id] = edge_ids_by_type
            self._boundary_edge_ids_by_partition_and_type[partition.partition_id] = boundary_edge_ids_by_type
            self._local_edge_id_by_partition_and_type[partition.partition_id] = local_edge_ids_by_type

    @classmethod
    def from_partition_dir(cls, root) -> "StoreBackedSamplingCoordinator":
        manifest, feature_store, graph_store = load_partitioned_stores(root)
        return cls(
            manifest=manifest,
            feature_store=feature_store,
            graph_store=graph_store,
        )

    def _collect_edge_types(self) -> tuple[tuple[str, str, str], ...]:
        ordered = []
        seen = set()
        for edge_type in self.graph_store.edge_types:
            edge_type = tuple(edge_type)
            if edge_type in seen:
                continue
            seen.add(edge_type)
            ordered.append(edge_type)
        for partition in self.manifest.partitions:
            for mapping in (partition.edge_ids_by_type, partition.boundary_edge_ids_by_type):
                for edge_type in mapping:
                    edge_type = tuple(edge_type)
                    if edge_type in seen:
                        continue
                    seen.add(edge_type)
                    ordered.append(edge_type)
        return tuple(ordered)

    def partition_ids(self) -> tuple[int, ...]:
        return tuple(partition.partition_id for partition in self.manifest.partitions)

    def _partition(self, partition_id: int) -> PartitionShard:
        try:
            return self._partitions[int(partition_id)]
        except KeyError as exc:
            raise KeyError(f"missing partition {partition_id}") from exc

    def partition_node_ids(self, partition_id: int, *, node_type: str = "node") -> torch.Tensor:
        start, end = self._partition(partition_id).node_range_for(node_type)
        return torch.arange(start, end, dtype=torch.long)

    def partition_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        edge_type = _resolve_edge_type(edge_type, self._edge_types)
        return torch.tensor(self._edge_ids_by_partition_and_type[int(partition_id)][edge_type], dtype=torch.long)

    def partition_boundary_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        edge_type = _resolve_edge_type(edge_type, self._edge_types)
        return torch.tensor(self._boundary_edge_ids_by_partition_and_type[int(partition_id)][edge_type], dtype=torch.long)

    def partition_incident_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        return torch.cat(
            (
                self.partition_edge_ids(partition_id, edge_type=edge_type),
                self.partition_boundary_edge_ids(partition_id, edge_type=edge_type),
            )
        )

    def fetch_partition_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
        global_ids: bool = False,
    ) -> torch.Tensor:
        edge_type = _resolve_edge_type(edge_type, self._edge_types)
        edge_index = self.graph_store.edge_index(edge_type, partition_id=partition_id)
        if not global_ids:
            return edge_index
        src_type, _, dst_type = edge_type
        partition = self._partition(partition_id)
        src_start, _ = partition.node_range_for(src_type)
        dst_start, _ = partition.node_range_for(dst_type)
        return torch.stack((edge_index[0] + src_start, edge_index[1] + dst_start))

    def fetch_partition_boundary_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
    ) -> torch.Tensor:
        edge_type = _resolve_edge_type(edge_type, self._edge_types)
        return self.graph_store.boundary_edge_index(edge_type, partition_id=partition_id)

    def fetch_partition_incident_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
    ) -> torch.Tensor:
        return torch.cat(
            (
                self.fetch_partition_edge_index(partition_id, edge_type=edge_type, global_ids=True),
                self.fetch_partition_boundary_edge_index(partition_id, edge_type=edge_type),
            ),
            dim=1,
        )

    def fetch_partition_adjacency(
        self,
        partition_id: int,
        *,
        edge_type=None,
        layout: SparseLayout | str = SparseLayout.COO,
    ) -> SparseTensor:
        edge_type = _resolve_edge_type(edge_type, self._edge_types)
        return self.graph_store.adjacency(edge_type=edge_type, layout=layout, partition_id=partition_id)

    def route_node_ids(self, node_ids: torch.Tensor, *, node_type: str = "node") -> tuple[ShardRoute, ...]:
        node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        grouped: dict[int, dict[str, list[int]]] = {}
        for position, node_id in enumerate(node_ids.tolist()):
            partition = self.manifest.owner(node_id, node_type=node_type)
            start, _ = partition.node_range_for(node_type)
            bucket = grouped.setdefault(
                partition.partition_id,
                {"global_ids": [], "local_ids": [], "positions": []},
            )
            bucket["global_ids"].append(int(node_id))
            bucket["local_ids"].append(int(node_id) - int(start))
            bucket["positions"].append(int(position))
        return _build_routes(grouped)

    def route_edge_ids(self, edge_ids: torch.Tensor, *, edge_type=None) -> tuple[ShardRoute, ...]:
        edge_type = _resolve_edge_type(edge_type, self._edge_types)
        try:
            owners = self._edge_partition_by_type[edge_type]
        except KeyError as exc:
            raise KeyError(edge_type) from exc

        edge_ids = torch.as_tensor(edge_ids, dtype=torch.long)
        grouped: dict[int, dict[str, list[int]]] = {}
        for position, edge_id in enumerate(edge_ids.tolist()):
            try:
                partition_id = owners[int(edge_id)]
            except KeyError as exc:
                raise KeyError(f"edge {edge_id} is not present for edge type {edge_type!r}") from exc
            local_edge_ids = self._local_edge_id_by_partition_and_type[partition_id][edge_type]
            bucket = grouped.setdefault(
                partition_id,
                {"global_ids": [], "local_ids": [], "positions": []},
            )
            bucket["global_ids"].append(int(edge_id))
            bucket["local_ids"].append(int(local_edge_ids[int(edge_id)]))
            bucket["positions"].append(int(position))
        return _build_routes(grouped)

    def fetch_node_features(self, key: FeatureKey, node_ids: torch.Tensor) -> TensorSlice:
        entity_kind, type_key, _ = key
        if entity_kind != "node":
            raise ValueError("fetch_node_features requires a node feature key")
        node_type = str(type_key)
        node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        routes = self.route_node_ids(node_ids, node_type=node_type)
        if node_ids.numel() == 0:
            shape = self.feature_store.shape(key)
            return TensorSlice(index=node_ids, values=torch.empty((0,) + shape[1:]))

        values = None
        for route in routes:
            fetched = self.feature_store.fetch(key, route.local_ids, partition_id=route.partition_id)
            if values is None:
                values = torch.empty(
                    (node_ids.numel(),) + tuple(fetched.values.shape[1:]),
                    dtype=fetched.values.dtype,
                    device=fetched.values.device,
                )
            values[route.positions] = fetched.values
        return TensorSlice(index=node_ids, values=values)

    def fetch_edge_features(self, key: FeatureKey, edge_ids: torch.Tensor) -> TensorSlice:
        entity_kind, type_key, _ = key
        if entity_kind != "edge":
            raise ValueError("fetch_edge_features requires an edge feature key")
        edge_type = tuple(type_key)
        edge_ids = torch.as_tensor(edge_ids, dtype=torch.long)
        if edge_ids.numel() == 0:
            shape = self.feature_store.shape(key)
            return TensorSlice(index=edge_ids, values=torch.empty((0,) + shape[1:]))

        grouped_local: dict[int, dict[str, list[int]]] = {}
        grouped_boundary: dict[int, dict[str, list[int]]] = {}
        owners = self._edge_partition_by_type.get(edge_type, {})
        boundary_owners = self._boundary_partition_by_type.get(edge_type, {})
        for position, edge_id in enumerate(edge_ids.tolist()):
            edge_id = int(edge_id)
            partition_id = owners.get(edge_id)
            if partition_id is not None:
                bucket = grouped_local.setdefault(
                    partition_id,
                    {"global_ids": [], "local_ids": [], "positions": []},
                )
                bucket["global_ids"].append(edge_id)
                bucket["local_ids"].append(
                    self._local_edge_id_by_partition_and_type[partition_id][edge_type][edge_id]
                )
                bucket["positions"].append(int(position))
                continue
            partition_id = boundary_owners.get(edge_id)
            if partition_id is not None:
                bucket = grouped_boundary.setdefault(
                    partition_id,
                    {"global_ids": [], "positions": []},
                )
                bucket["global_ids"].append(edge_id)
                bucket["positions"].append(int(position))
                continue
            raise KeyError(f"edge {edge_id} is not present for edge type {edge_type!r}")

        values = None
        for route in _build_routes(grouped_local):
            fetched = self.feature_store.fetch(key, route.local_ids, partition_id=route.partition_id)
            if values is None:
                values = torch.empty(
                    (edge_ids.numel(),) + tuple(fetched.values.shape[1:]),
                    dtype=fetched.values.dtype,
                    device=fetched.values.device,
                )
            values[route.positions] = fetched.values
        for partition_id in sorted(grouped_boundary):
            bucket = grouped_boundary[partition_id]
            fetched = self.feature_store.fetch_boundary(
                key,
                torch.tensor(bucket["global_ids"], dtype=torch.long),
                partition_id=partition_id,
            )
            if values is None:
                values = torch.empty(
                    (edge_ids.numel(),) + tuple(fetched.values.shape[1:]),
                    dtype=fetched.values.dtype,
                    device=fetched.values.device,
                )
            values[torch.tensor(bucket["positions"], dtype=torch.long)] = fetched.values
        return TensorSlice(index=edge_ids, values=values)


__all__ = [
    "LocalSamplingCoordinator",
    "SamplingCoordinator",
    "ShardRoute",
    "StoreBackedSamplingCoordinator",
]
