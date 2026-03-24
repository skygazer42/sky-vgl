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

    def adjacency(
        self,
        *,
        edge_type: EdgeType | None = None,
        layout: SparseLayout | str = SparseLayout.COO,
        partition_id: int | None = None,
    ) -> SparseTensor:
        ...


class LocalFeatureStoreAdapter:
    def __init__(self, store: FeatureStore):
        self._store = store

    def fetch(
        self,
        key: FeatureKey,
        index: torch.Tensor,
        *,
        partition_id: int | None = None,
    ) -> TensorSlice:
        return self._store.fetch(key, index)

    def shape(
        self,
        key: FeatureKey,
        *,
        partition_id: int | None = None,
    ) -> tuple[int, ...]:
        return self._store.shape(key)


class LocalGraphStoreAdapter:
    def __init__(self, store: GraphStore):
        self._store = store

    @property
    def edge_types(self) -> tuple[EdgeType, ...]:
        return self._store.edge_types

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

    def adjacency(
        self,
        *,
        edge_type: EdgeType | None = None,
        layout: SparseLayout | str = SparseLayout.COO,
        partition_id: int | None = None,
    ) -> SparseTensor:
        return self._store.adjacency(edge_type=edge_type, layout=layout)


__all__ = [
    "DistributedFeatureStore",
    "DistributedGraphStore",
    "LocalFeatureStoreAdapter",
    "LocalGraphStoreAdapter",
]
