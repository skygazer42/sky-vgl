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


def _tensor_ints(values: torch.Tensor) -> tuple[int, ...]:
    tensor = torch.as_tensor(values, dtype=torch.long).view(-1)
    return tuple(int(value.item()) for value in tensor)


def _build_int_lookup(mapping: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if not mapping:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    keys = torch.tensor(sorted(mapping), dtype=torch.long)
    values = torch.tensor([mapping[int(key)] for key in keys], dtype=torch.long)
    return keys, values


def _lookup_tensor_values(index_ids: torch.Tensor, query_ids: torch.Tensor, lookup_values: torch.Tensor, *, entity_name: str) -> torch.Tensor:
    index_ids = torch.as_tensor(index_ids, dtype=torch.long).view(-1)
    query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=index_ids.device).view(-1)
    lookup_values = torch.as_tensor(lookup_values, dtype=torch.long, device=index_ids.device).view(-1)
    if query_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=index_ids.device)
    if index_ids.numel() == 0:
        raise KeyError(f"{entity_name} {int(query_ids[0])} is not present")

    positions = torch.searchsorted(index_ids, query_ids.contiguous())
    missing = positions >= index_ids.numel()
    if bool(missing.any()):
        raise KeyError(f"{entity_name} {int(query_ids[missing][0])} is not present")
    matched = index_ids.index_select(0, positions)
    mismatch = matched != query_ids
    if bool(mismatch.any()):
        raise KeyError(f"{entity_name} {int(query_ids[mismatch][0])} is not present")
    return lookup_values.index_select(0, positions)


def _optional_lookup_tensor_values(index_ids: torch.Tensor, query_ids: torch.Tensor, lookup_values: torch.Tensor) -> torch.Tensor:
    index_ids = torch.as_tensor(index_ids, dtype=torch.long).view(-1)
    query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=index_ids.device).view(-1)
    lookup_values = torch.as_tensor(lookup_values, dtype=torch.long, device=index_ids.device).view(-1)
    if query_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=index_ids.device)
    if index_ids.numel() == 0:
        return torch.full_like(query_ids, -1)

    positions = torch.searchsorted(index_ids, query_ids.contiguous())
    matched = positions < index_ids.numel()
    if bool(matched.any()):
        valid = torch.zeros_like(matched)
        valid[matched] = index_ids.index_select(0, positions[matched]) == query_ids[matched]
        matched = valid
    result = torch.full_like(query_ids, -1)
    if bool(matched.any()):
        result[matched] = lookup_values.index_select(0, positions[matched])
    return result


def _group_partitioned_ids(
    global_ids: torch.Tensor,
    partition_ids: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
) -> tuple[tuple[int, torch.Tensor, torch.Tensor], ...]:
    global_ids = torch.as_tensor(global_ids, dtype=torch.long).view(-1)
    partition_ids = torch.as_tensor(partition_ids, dtype=torch.long, device=global_ids.device).view(-1)
    if global_ids.numel() == 0:
        return ()
    if positions is None:
        positions = torch.arange(global_ids.numel(), dtype=torch.long, device=global_ids.device)
    else:
        positions = torch.as_tensor(positions, dtype=torch.long, device=global_ids.device).view(-1)
    groups = []
    for partition_id_tensor in torch.unique(partition_ids, sorted=True):
        partition_id = int(partition_id_tensor)
        mask = partition_ids == partition_id_tensor
        groups.append((partition_id, global_ids[mask], positions[mask]))
    return tuple(groups)


def _build_node_route_lookup(manifest: PartitionManifest) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    lookup: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for node_type in manifest.num_nodes_by_type:
        ranges = []
        for partition in manifest.partitions:
            if node_type not in partition.node_ranges:
                continue
            start, end = partition.node_range_for(node_type)
            if end <= start:
                continue
            ranges.append((int(start), int(end), int(partition.partition_id)))
        ranges.sort(key=lambda item: item[0])
        if not ranges:
            empty = torch.empty(0, dtype=torch.long)
            lookup[node_type] = (empty, empty, empty)
            continue
        starts = torch.tensor([start for start, _, _ in ranges], dtype=torch.long)
        ends = torch.tensor([end for _, end, _ in ranges], dtype=torch.long)
        partition_ids = torch.tensor([partition_id for _, _, partition_id in ranges], dtype=torch.long)
        lookup[node_type] = (starts, ends, partition_ids)
    return lookup


def _route_node_ids_with_lookup(
    manifest: PartitionManifest,
    lookup: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    node_ids: torch.Tensor,
    *,
    node_type: str = "node",
) -> tuple[ShardRoute, ...]:
    node_type = str(node_type)
    node_ids = torch.as_tensor(node_ids, dtype=torch.long)
    if node_ids.numel() == 0:
        return ()
    try:
        typed_count = int(manifest.num_nodes_by_type[node_type])
        starts, ends, partition_ids = lookup[node_type]
    except KeyError as exc:
        raise KeyError(node_type) from exc

    invalid_range = (node_ids < 0) | (node_ids >= typed_count)
    if bool(invalid_range.any()):
        raise KeyError(int(node_ids[invalid_range][0].item()))
    if ends.numel() == 0:
        raise KeyError(int(node_ids[0].item()))

    route_index = torch.searchsorted(ends, node_ids, right=True)
    invalid_partition = route_index >= ends.numel()
    if bool(invalid_partition.any()):
        raise KeyError(int(node_ids[invalid_partition][0].item()))
    local_starts = starts.index_select(0, route_index)
    valid = local_starts <= node_ids
    if bool((~valid).any()):
        raise KeyError(int(node_ids[~valid][0].item()))

    local_ids = node_ids - local_starts
    routed_partition_ids = partition_ids.index_select(0, route_index)
    positions = torch.arange(node_ids.numel(), dtype=torch.long)
    routes = []
    for partition_id_tensor in torch.unique(routed_partition_ids, sorted=True):
        partition_id = int(partition_id_tensor.item())
        partition_mask = routed_partition_ids == partition_id
        partition_positions = positions[partition_mask]
        routes.append(
            ShardRoute(
                partition_id=partition_id,
                global_ids=node_ids[partition_positions],
                local_ids=local_ids[partition_positions],
                positions=partition_positions,
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
            partition_id: LocalFeatureStoreAdapter(
                shard.feature_store,
                boundary_edge_data_by_type=shard.boundary_edge_data_by_type,
            )
            for partition_id, shard in self.shards.items()
        }
        self._node_route_lookup = _build_node_route_lookup(self.manifest)
        self._edge_partition_by_type: dict[object, dict[int, int]] = {}
        self._boundary_partition_by_type: dict[object, dict[int, int]] = {}
        self._edge_owner_lookup_by_type: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        self._boundary_owner_lookup_by_type: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        for partition_id, shard in self.shards.items():
            for edge_type in shard.graph.edges:
                owner = self._edge_partition_by_type.setdefault(edge_type, {})
                boundary_owner = self._boundary_partition_by_type.setdefault(edge_type, {})
                for edge_id in _tensor_ints(shard.edge_ids(edge_type=edge_type)):
                    if edge_id in owner:
                        raise ValueError(f"duplicate global edge id {edge_id} for edge type {edge_type!r}")
                    owner[edge_id] = partition_id
                for edge_id in _tensor_ints(shard.boundary_edge_ids(edge_type=edge_type)):
                    boundary_owner.setdefault(edge_id, partition_id)
        for edge_type, owner in self._edge_partition_by_type.items():
            self._edge_owner_lookup_by_type[edge_type] = _build_int_lookup(owner)
        for edge_type, owner in self._boundary_partition_by_type.items():
            self._boundary_owner_lookup_by_type[edge_type] = _build_int_lookup(owner)

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
        routes = _route_node_ids_with_lookup(
            self.manifest,
            self._node_route_lookup,
            node_ids,
            node_type=node_type,
        )
        for route in routes:
            if route.partition_id not in self.shards:
                raise KeyError(f"missing shard for partition {route.partition_id}")
        return routes

    def route_edge_ids(self, edge_ids: torch.Tensor, *, edge_type=None) -> tuple[ShardRoute, ...]:
        edge_type = _resolve_edge_type(edge_type, next(iter(self.shards.values())).graph.edges)
        try:
            owner_ids, owner_partitions = self._edge_owner_lookup_by_type[edge_type]
        except KeyError as exc:
            raise KeyError(edge_type) from exc

        edge_ids = torch.as_tensor(edge_ids, dtype=torch.long)
        routed_partition_ids = _lookup_tensor_values(owner_ids, edge_ids, owner_partitions, entity_name="edge")
        routes = []
        for partition_id, global_ids, positions in _group_partitioned_ids(edge_ids, routed_partition_ids):
            shard = self.shards.get(partition_id)
            if shard is None:
                raise KeyError(f"missing shard for partition {partition_id}")
            routes.append(
                ShardRoute(
                    partition_id=partition_id,
                    global_ids=global_ids,
                    local_ids=shard.global_to_local_edge(global_ids, edge_type=edge_type),
                    positions=positions,
                )
            )
        return tuple(routes)

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
        if edge_ids.numel() == 0:
            first_store = next(iter(self.feature_stores.values()))
            shape = first_store.shape(key)
            return TensorSlice(index=edge_ids, values=torch.empty((0,) + shape[1:]))

        owner_ids, owner_partitions = self._edge_owner_lookup_by_type.get(
            edge_type,
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        )
        boundary_ids, boundary_partitions = self._boundary_owner_lookup_by_type.get(
            edge_type,
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        )
        local_partition_ids = _optional_lookup_tensor_values(owner_ids, edge_ids, owner_partitions)
        local_mask = local_partition_ids >= 0
        boundary_partition_ids = _optional_lookup_tensor_values(boundary_ids, edge_ids, boundary_partitions)
        boundary_mask = (~local_mask) & (boundary_partition_ids >= 0)
        unresolved_mask = (~local_mask) & (~boundary_mask)
        if bool(unresolved_mask.any()):
            missing_edge_id = int(edge_ids[unresolved_mask][0])
            raise KeyError(f"edge {missing_edge_id} is not present for edge type {edge_type!r}")

        values = None
        all_positions = torch.arange(edge_ids.numel(), dtype=torch.long, device=edge_ids.device)
        for partition_id, global_ids, positions in _group_partitioned_ids(
            edge_ids[local_mask],
            local_partition_ids[local_mask],
            positions=all_positions[local_mask],
        ):
            route = ShardRoute(
                partition_id=partition_id,
                global_ids=global_ids,
                local_ids=self._shard(partition_id).global_to_local_edge(global_ids, edge_type=edge_type),
                positions=positions,
            )
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
        for partition_id, global_ids, positions in _group_partitioned_ids(
            edge_ids[boundary_mask],
            boundary_partition_ids[boundary_mask],
            positions=all_positions[boundary_mask],
        ):
            fetched = self.feature_stores[partition_id].fetch_boundary(
                key,
                global_ids,
                partition_id=partition_id,
            )
            if values is None:
                values = torch.empty(
                    (edge_ids.numel(),) + tuple(fetched.values.shape[1:]),
                    dtype=fetched.values.dtype,
                    device=fetched.values.device,
                )
            values[positions] = fetched.values
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
        self._edge_owner_lookup_by_type: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._boundary_owner_lookup_by_type: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._local_edge_lookup_by_partition_and_type: dict[int, dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]] = {}
        self._node_route_lookup = _build_node_route_lookup(manifest)

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
            self._local_edge_lookup_by_partition_and_type[partition.partition_id] = {
                edge_type: (
                    torch.tensor(edge_ids_by_type[edge_type], dtype=torch.long),
                    torch.arange(len(edge_ids_by_type[edge_type]), dtype=torch.long),
                )
                for edge_type in self._edge_types
            }
        for edge_type, owner in self._edge_partition_by_type.items():
            self._edge_owner_lookup_by_type[edge_type] = _build_int_lookup(owner)
        for edge_type, owner in self._boundary_partition_by_type.items():
            self._boundary_owner_lookup_by_type[edge_type] = _build_int_lookup(owner)

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
        return _route_node_ids_with_lookup(
            self.manifest,
            self._node_route_lookup,
            node_ids,
            node_type=node_type,
        )

    def route_edge_ids(self, edge_ids: torch.Tensor, *, edge_type=None) -> tuple[ShardRoute, ...]:
        edge_type = _resolve_edge_type(edge_type, self._edge_types)
        try:
            owner_ids, owner_partitions = self._edge_owner_lookup_by_type[edge_type]
        except KeyError as exc:
            raise KeyError(edge_type) from exc

        edge_ids = torch.as_tensor(edge_ids, dtype=torch.long)
        routed_partition_ids = _lookup_tensor_values(owner_ids, edge_ids, owner_partitions, entity_name="edge")
        routes = []
        for partition_id, global_ids, positions in _group_partitioned_ids(edge_ids, routed_partition_ids):
            lookup_ids, lookup_local_ids = self._local_edge_lookup_by_partition_and_type[partition_id][edge_type]
            routes.append(
                ShardRoute(
                    partition_id=partition_id,
                    global_ids=global_ids,
                    local_ids=_lookup_tensor_values(lookup_ids, global_ids, lookup_local_ids, entity_name="edge"),
                    positions=positions,
                )
            )
        return tuple(routes)

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

        owner_ids, owner_partitions = self._edge_owner_lookup_by_type.get(
            edge_type,
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        )
        boundary_ids, boundary_partitions = self._boundary_owner_lookup_by_type.get(
            edge_type,
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        )
        local_partition_ids = _optional_lookup_tensor_values(owner_ids, edge_ids, owner_partitions)
        local_mask = local_partition_ids >= 0
        boundary_partition_ids = _optional_lookup_tensor_values(boundary_ids, edge_ids, boundary_partitions)
        boundary_mask = (~local_mask) & (boundary_partition_ids >= 0)
        unresolved_mask = (~local_mask) & (~boundary_mask)
        if bool(unresolved_mask.any()):
            missing_edge_id = int(edge_ids[unresolved_mask][0])
            raise KeyError(f"edge {missing_edge_id} is not present for edge type {edge_type!r}")

        values = None
        all_positions = torch.arange(edge_ids.numel(), dtype=torch.long, device=edge_ids.device)
        for partition_id, global_ids, positions in _group_partitioned_ids(
            edge_ids[local_mask],
            local_partition_ids[local_mask],
            positions=all_positions[local_mask],
        ):
            lookup_ids, lookup_local_ids = self._local_edge_lookup_by_partition_and_type[partition_id][edge_type]
            route = ShardRoute(
                partition_id=partition_id,
                global_ids=global_ids,
                local_ids=_lookup_tensor_values(lookup_ids, global_ids, lookup_local_ids, entity_name="edge"),
                positions=positions,
            )
            fetched = self.feature_store.fetch(key, route.local_ids, partition_id=route.partition_id)
            if values is None:
                values = torch.empty(
                    (edge_ids.numel(),) + tuple(fetched.values.shape[1:]),
                    dtype=fetched.values.dtype,
                    device=fetched.values.device,
                )
            values[route.positions] = fetched.values
        for partition_id, global_ids, positions in _group_partitioned_ids(
            edge_ids[boundary_mask],
            boundary_partition_ids[boundary_mask],
            positions=all_positions[boundary_mask],
        ):
            fetched = self.feature_store.fetch_boundary(
                key,
                global_ids,
                partition_id=partition_id,
            )
            if values is None:
                values = torch.empty(
                    (edge_ids.numel(),) + tuple(fetched.values.shape[1:]),
                    dtype=fetched.values.dtype,
                    device=fetched.values.device,
                )
            values[positions] = fetched.values
        return TensorSlice(index=edge_ids, values=values)


__all__ = [
    "LocalSamplingCoordinator",
    "SamplingCoordinator",
    "ShardRoute",
    "StoreBackedSamplingCoordinator",
]
