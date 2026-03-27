from collections.abc import Mapping
from typing import Protocol

import torch

from vgl.sparse import SparseLayout, SparseTensor
from vgl.storage import FeatureStore, GraphStore
from vgl.storage.base import TensorSlice
from vgl.storage.feature_store import FeatureKey
from vgl.storage.graph_store import EdgeType


class DistributedFeatureStore(Protocol):
    def fetch(
        self,
        key: FeatureKey,
        index: torch.Tensor,
        *,
        partition_id: int | None = None,
    ) -> TensorSlice:
        ...

    def fetch_boundary(
        self,
        key: FeatureKey,
        index: torch.Tensor,
        *,
        partition_id: int | None = None,
    ) -> TensorSlice:
        ...

    def shape(
        self,
        key: FeatureKey,
        *,
        partition_id: int | None = None,
    ) -> tuple[int, ...]:
        ...


class DistributedGraphStore(Protocol):
    @property
    def edge_types(self) -> tuple[EdgeType, ...]:
        ...

    def num_nodes(
        self,
        node_type: str = "node",
        *,
        partition_id: int | None = None,
    ) -> int:
        ...

    def edge_index(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> torch.Tensor:
        ...

    def edge_count(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> int:
        ...

    def boundary_edge_index(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> torch.Tensor:
        ...

    def adjacency(
        self,
        *,
        edge_type: EdgeType | None = None,
        layout: SparseLayout | str = SparseLayout.COO,
        partition_id: int | None = None,
    ) -> SparseTensor:
        ...


class LocalFeatureStoreAdapter:
    def __init__(self, store: FeatureStore, *, boundary_edge_data_by_type=None):
        self._store = store
        self._boundary_edge_data_by_type = {
            tuple(edge_type): {
                key: value
                for key, value in dict(edge_data).items()
            }
            for edge_type, edge_data in dict(boundary_edge_data_by_type or {}).items()
        }
        self._boundary_edge_local_id_by_type = {
            edge_type: {
                int(edge_id): index
                for index, edge_id in enumerate(
                    torch.as_tensor(edge_data.get("e_id", torch.empty((0,), dtype=torch.long)), dtype=torch.long).tolist()
                )
            }
            for edge_type, edge_data in self._boundary_edge_data_by_type.items()
        }

    def fetch(
        self,
        key: FeatureKey,
        index: torch.Tensor,
        *,
        partition_id: int | None = None,
    ) -> TensorSlice:
        return self._store.fetch(key, index)

    def fetch_boundary(
        self,
        key: FeatureKey,
        index: torch.Tensor,
        *,
        partition_id: int | None = None,
    ) -> TensorSlice:
        entity_kind, type_key, feature_name = key
        if entity_kind != "edge":
            raise ValueError("fetch_boundary requires an edge feature key")
        edge_type = tuple(type_key)
        try:
            edge_data = self._boundary_edge_data_by_type[edge_type]
            positions = self._boundary_edge_local_id_by_type[edge_type]
        except KeyError as exc:
            raise KeyError(key) from exc
        feature = edge_data.get(feature_name)
        if not isinstance(feature, torch.Tensor):
            raise KeyError(key)

        index = torch.as_tensor(index, dtype=torch.long).view(-1)
        if index.numel() == 0:
            return TensorSlice(index=index, values=feature.new_empty((0,) + tuple(feature.shape[1:])))
        try:
            local_ids = [positions[int(edge_id)] for edge_id in index.tolist()]
        except KeyError as exc:
            raise KeyError(key) from exc
        return TensorSlice(index=index, values=feature[torch.tensor(local_ids, dtype=torch.long)])

    def shape(
        self,
        key: FeatureKey,
        *,
        partition_id: int | None = None,
    ) -> tuple[int, ...]:
        return self._store.shape(key)


class LocalGraphStoreAdapter:
    def __init__(
        self,
        store: GraphStore,
        *,
        boundary_edge_index_by_type: Mapping[EdgeType, torch.Tensor] | None = None,
    ):
        self._store = store
        self._boundary_edge_index_by_type = {
            tuple(edge_type): torch.as_tensor(edge_index, dtype=torch.long)
            for edge_type, edge_index in dict(boundary_edge_index_by_type or {}).items()
        }

    def _resolve_edge_type(self, edge_type: EdgeType | None) -> EdgeType:
        if edge_type is not None:
            return tuple(edge_type)
        default_edge_type = ("node", "to", "node")
        if default_edge_type in self._store.edge_types:
            return default_edge_type
        if len(self._store.edge_types) == 1:
            return next(iter(self._store.edge_types))
        raise KeyError("edge_type is required when multiple edge types exist")

    @property
    def edge_types(self) -> tuple[EdgeType, ...]:
        return self._store.edge_types

    def num_nodes(
        self,
        node_type: str = "node",
        *,
        partition_id: int | None = None,
    ) -> int:
        return self._store.num_nodes(node_type)

    def edge_index(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> torch.Tensor:
        return self._store.edge_index(edge_type)

    def edge_count(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> int:
        return self._store.edge_count(edge_type)

    def boundary_edge_index(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> torch.Tensor:
        resolved = self._resolve_edge_type(edge_type)
        boundary_edge_index = self._boundary_edge_index_by_type.get(resolved)
        if boundary_edge_index is not None:
            return boundary_edge_index
        reference = self._store.edge_index(resolved)
        return torch.empty((2, 0), dtype=torch.long, device=reference.device)

    def adjacency(
        self,
        *,
        edge_type: EdgeType | None = None,
        layout: SparseLayout | str = SparseLayout.COO,
        partition_id: int | None = None,
    ) -> SparseTensor:
        return self._store.adjacency(edge_type=edge_type, layout=layout)


class PartitionedFeatureStore:
    def __init__(self, stores: Mapping[int, DistributedFeatureStore]):
        self._stores = {int(partition_id): store for partition_id, store in dict(stores).items()}
        if not self._stores:
            raise ValueError("PartitionedFeatureStore requires at least one partition store")

    def _store(self, partition_id: int | None) -> DistributedFeatureStore:
        if partition_id is None:
            if len(self._stores) == 1:
                return next(iter(self._stores.values()))
            raise ValueError("partition_id is required when multiple partition stores exist")
        try:
            return self._stores[int(partition_id)]
        except KeyError as exc:
            raise KeyError(f"missing feature store for partition {partition_id}") from exc

    def fetch(
        self,
        key: FeatureKey,
        index: torch.Tensor,
        *,
        partition_id: int | None = None,
    ) -> TensorSlice:
        return self._store(partition_id).fetch(key, index, partition_id=partition_id)

    def fetch_boundary(
        self,
        key: FeatureKey,
        index: torch.Tensor,
        *,
        partition_id: int | None = None,
    ) -> TensorSlice:
        return self._store(partition_id).fetch_boundary(key, index, partition_id=partition_id)

    def shape(
        self,
        key: FeatureKey,
        *,
        partition_id: int | None = None,
    ) -> tuple[int, ...]:
        if partition_id is not None:
            return self._store(partition_id).shape(key, partition_id=partition_id)
        first_error = None
        for current_partition_id, store in self._stores.items():
            try:
                return store.shape(key, partition_id=current_partition_id)
            except KeyError as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        raise KeyError(key)


class PartitionedGraphStore:
    def __init__(self, stores: Mapping[int, DistributedGraphStore]):
        self._stores = {int(partition_id): store for partition_id, store in dict(stores).items()}
        if not self._stores:
            raise ValueError("PartitionedGraphStore requires at least one partition store")

    def _store(self, partition_id: int | None) -> DistributedGraphStore:
        if partition_id is None:
            if len(self._stores) == 1:
                return next(iter(self._stores.values()))
            raise ValueError("partition_id is required when multiple partition stores exist")
        try:
            return self._stores[int(partition_id)]
        except KeyError as exc:
            raise KeyError(f"missing graph store for partition {partition_id}") from exc

    @property
    def edge_types(self) -> tuple[EdgeType, ...]:
        ordered = []
        seen = set()
        for partition_id in sorted(self._stores):
            for edge_type in self._stores[partition_id].edge_types:
                edge_type = tuple(edge_type)
                if edge_type in seen:
                    continue
                seen.add(edge_type)
                ordered.append(edge_type)
        return tuple(ordered)

    def num_nodes(
        self,
        node_type: str = "node",
        *,
        partition_id: int | None = None,
    ) -> int:
        return self._store(partition_id).num_nodes(node_type, partition_id=partition_id)

    def edge_index(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> torch.Tensor:
        return self._store(partition_id).edge_index(edge_type, partition_id=partition_id)

    def edge_count(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> int:
        return self._store(partition_id).edge_count(edge_type, partition_id=partition_id)

    def boundary_edge_index(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> torch.Tensor:
        return self._store(partition_id).boundary_edge_index(edge_type, partition_id=partition_id)

    def adjacency(
        self,
        *,
        edge_type: EdgeType | None = None,
        layout: SparseLayout | str = SparseLayout.COO,
        partition_id: int | None = None,
    ) -> SparseTensor:
        return self._store(partition_id).adjacency(
            edge_type=edge_type,
            layout=layout,
            partition_id=partition_id,
        )


__all__ = [
    "DistributedFeatureStore",
    "DistributedGraphStore",
    "LocalFeatureStoreAdapter",
    "LocalGraphStoreAdapter",
    "PartitionedFeatureStore",
    "PartitionedGraphStore",
]
