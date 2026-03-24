from dataclasses import dataclass
from typing import Protocol

import torch

from vgl.distributed.shard import LocalGraphShard
from vgl.distributed.store import DistributedFeatureStore, LocalFeatureStoreAdapter
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

    def fetch_partition_edge_index(
        self,
        partition_id: int,
        *,
        edge_type=None,
        global_ids: bool = False,
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

    def _shard(self, partition_id: int) -> LocalGraphShard:
        try:
            return self.shards[int(partition_id)]
        except KeyError as exc:
            raise KeyError(f"missing shard for partition {partition_id}") from exc

    def partition_node_ids(self, partition_id: int, *, node_type: str = "node") -> torch.Tensor:
        return self._shard(partition_id).node_ids_for(node_type)

    def partition_edge_ids(self, partition_id: int, *, edge_type=None) -> torch.Tensor:
        return self._shard(partition_id).edge_ids(edge_type=edge_type)

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

    def route_edge_ids(self, edge_ids: torch.Tensor, *, edge_type=None) -> tuple[ShardRoute, ...]:
        if edge_type is None:
            first_shard = next(iter(self.shards.values()))
            edge_type = first_shard.graph._default_edge_type()
        edge_type = tuple(edge_type)
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


__all__ = ["LocalSamplingCoordinator", "SamplingCoordinator", "ShardRoute"]
