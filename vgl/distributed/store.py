from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from vgl.distributed.partition import PartitionManifest, PartitionShard, load_partition_manifest
from vgl.sparse import SparseLayout, SparseTensor
from vgl.storage import FeatureStore, GraphStore, InMemoryGraphStore, InMemoryTensorStore
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


def _partition_edge_types(partition: PartitionShard) -> tuple[EdgeType, ...]:
    ordered = []
    seen = set()
    for mapping in (partition.edge_ids_by_type, partition.boundary_edge_ids_by_type):
        for edge_type in mapping:
            edge_type = tuple(edge_type)
            if edge_type in seen:
                continue
            seen.add(edge_type)
            ordered.append(edge_type)
    return tuple(ordered)


@dataclass(slots=True)
class _PartitionStoreBundle:
    feature_store: LocalFeatureStoreAdapter
    graph_store: LocalGraphStoreAdapter


class _PartitionStoreBundleCache:
    def __init__(self, root: Path, partitions: Mapping[int, PartitionShard]):
        self._root = Path(root)
        self._partitions = {int(partition_id): partition for partition_id, partition in dict(partitions).items()}
        self._bundles: dict[int, _PartitionStoreBundle] = {}

    def _partition(self, partition_id: int) -> PartitionShard:
        try:
            return self._partitions[int(partition_id)]
        except KeyError as exc:
            raise KeyError(f"missing partition {partition_id}") from exc

    def bundle(self, partition_id: int) -> _PartitionStoreBundle:
        partition_id = int(partition_id)
        bundle = self._bundles.get(partition_id)
        if bundle is not None:
            return bundle

        partition = self._partition(partition_id)
        payload = _partition_payload(self._root, partition)
        bundle = _PartitionStoreBundle(
            feature_store=_feature_store_adapter_from_payload(payload),
            graph_store=_graph_store_adapter_from_payload(payload, partition),
        )
        self._bundles[partition_id] = bundle
        return bundle


class _LazyPartitionFeatureStoreAdapter:
    def __init__(self, cache: _PartitionStoreBundleCache, partition_id: int):
        self._cache = cache
        self._partition_id = int(partition_id)

    @property
    def _store(self) -> LocalFeatureStoreAdapter:
        return self._cache.bundle(self._partition_id).feature_store

    def fetch(
        self,
        key: FeatureKey,
        index: torch.Tensor,
        *,
        partition_id: int | None = None,
    ) -> TensorSlice:
        return self._store.fetch(key, index, partition_id=self._partition_id if partition_id is None else partition_id)

    def fetch_boundary(
        self,
        key: FeatureKey,
        index: torch.Tensor,
        *,
        partition_id: int | None = None,
    ) -> TensorSlice:
        return self._store.fetch_boundary(
            key,
            index,
            partition_id=self._partition_id if partition_id is None else partition_id,
        )

    def shape(
        self,
        key: FeatureKey,
        *,
        partition_id: int | None = None,
    ) -> tuple[int, ...]:
        return self._store.shape(key, partition_id=self._partition_id if partition_id is None else partition_id)


class _LazyPartitionGraphStoreAdapter:
    def __init__(self, cache: _PartitionStoreBundleCache, partition: PartitionShard):
        self._cache = cache
        self._partition_id = int(partition.partition_id)
        self._edge_types = _partition_edge_types(partition)
        self._edge_ids_by_type = {
            tuple(edge_type): tuple(int(edge_id) for edge_id in edge_ids)
            for edge_type, edge_ids in partition.edge_ids_by_type.items()
        }
        self._num_nodes_by_type = {
            str(node_type): int(end) - int(start)
            for node_type, (start, end) in partition.node_ranges.items()
        }

    @property
    def _store(self) -> LocalGraphStoreAdapter:
        return self._cache.bundle(self._partition_id).graph_store

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
    def edge_types(self) -> tuple[EdgeType, ...]:
        return self._edge_types

    def num_nodes(
        self,
        node_type: str = "node",
        *,
        partition_id: int | None = None,
    ) -> int:
        try:
            return self._num_nodes_by_type[str(node_type)]
        except KeyError as exc:
            raise KeyError(node_type) from exc

    def edge_index(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> torch.Tensor:
        return self._store.edge_index(
            edge_type,
            partition_id=self._partition_id if partition_id is None else partition_id,
        )

    def edge_count(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> int:
        resolved = self._resolve_edge_type(edge_type)
        return len(self._edge_ids_by_type.get(resolved, ()))

    def boundary_edge_index(
        self,
        edge_type: EdgeType | None = None,
        *,
        partition_id: int | None = None,
    ) -> torch.Tensor:
        return self._store.boundary_edge_index(
            edge_type,
            partition_id=self._partition_id if partition_id is None else partition_id,
        )

    def adjacency(
        self,
        *,
        edge_type: EdgeType | None = None,
        layout: SparseLayout | str = SparseLayout.COO,
        partition_id: int | None = None,
    ) -> SparseTensor:
        return self._store.adjacency(
            edge_type=edge_type,
            layout=layout,
            partition_id=self._partition_id if partition_id is None else partition_id,
        )


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


def _partition_payload(root, partition: PartitionShard) -> dict:
    if partition.path is None:
        raise ValueError(f"partition {partition.partition_id} does not declare a payload path")
    return torch.load(Path(root) / partition.path, weights_only=True)


def _node_ids_by_type(payload: dict, partition: PartitionShard) -> dict[str, torch.Tensor]:
    graph_payload = payload["graph"]
    node_payloads = {
        str(node_type): dict(node_data)
        for node_type, node_data in graph_payload["nodes"].items()
    }
    raw_node_ids = payload.get("node_ids")
    node_ids_by_type = {}
    if isinstance(raw_node_ids, dict):
        node_ids_by_type = {
            str(node_type): torch.as_tensor(node_ids, dtype=torch.long)
            for node_type, node_ids in raw_node_ids.items()
        }
    elif raw_node_ids is not None and len(node_payloads) == 1:
        only_node_type = next(iter(node_payloads))
        node_ids_by_type[only_node_type] = torch.as_tensor(raw_node_ids, dtype=torch.long)

    for node_type, node_data in node_payloads.items():
        if node_type in node_ids_by_type:
            continue
        node_ids = node_data.get("n_id")
        if node_ids is None:
            start, end = partition.node_range_for(node_type)
            node_ids = torch.arange(start, end, dtype=torch.long)
        node_ids_by_type[node_type] = torch.as_tensor(node_ids, dtype=torch.long)
    return node_ids_by_type


def _boundary_edge_data_by_type(payload: dict, edge_types) -> dict[EdgeType, dict]:
    raw_boundary_edges = payload.get("boundary_edges", {})
    boundary_edge_data_by_type = {}
    for edge_type in edge_types:
        edge_type = tuple(edge_type)
        raw_edge_data = raw_boundary_edges.get(edge_type)
        if raw_edge_data is None:
            raw_edge_data = {
                "edge_index": torch.empty((2, 0), dtype=torch.long),
                "e_id": torch.empty((0,), dtype=torch.long),
            }
        boundary_edge_data_by_type[edge_type] = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in dict(raw_edge_data).items()
        }
        boundary_edge_data_by_type[edge_type].setdefault(
            "edge_index",
            torch.empty((2, 0), dtype=torch.long),
        )
        boundary_edge_data_by_type[edge_type].setdefault(
            "e_id",
            torch.empty((0,), dtype=torch.long),
        )
    return boundary_edge_data_by_type


def _feature_store_adapter_from_payload(payload: dict) -> LocalFeatureStoreAdapter:
    graph_payload = payload["graph"]
    node_payloads = {
        str(node_type): dict(node_data)
        for node_type, node_data in graph_payload["nodes"].items()
    }
    edge_payloads = {
        tuple(edge_type): dict(edge_data)
        for edge_type, edge_data in graph_payload["edges"].items()
    }
    edge_types = tuple(edge_payloads)
    feature_store = FeatureStore(
        {
            **{
                ("node", node_type, name): InMemoryTensorStore(value)
                for node_type, node_data in node_payloads.items()
                for name, value in node_data.items()
                if isinstance(value, torch.Tensor)
            },
            **{
                ("edge", edge_type, name): InMemoryTensorStore(value)
                for edge_type, edge_data in edge_payloads.items()
                for name, value in edge_data.items()
                if name != "edge_index" and isinstance(value, torch.Tensor)
            },
        }
    )
    return LocalFeatureStoreAdapter(
        feature_store,
        boundary_edge_data_by_type=_boundary_edge_data_by_type(payload, edge_types),
    )


def _graph_store_adapter_from_payload(payload: dict, partition: PartitionShard) -> LocalGraphStoreAdapter:
    graph_payload = payload["graph"]
    edge_payloads = {
        tuple(edge_type): dict(edge_data)
        for edge_type, edge_data in graph_payload["edges"].items()
    }
    node_ids_by_type = _node_ids_by_type(payload, partition)
    graph_store = InMemoryGraphStore(
        edges={
            edge_type: torch.as_tensor(edge_payload["edge_index"], dtype=torch.long)
            for edge_type, edge_payload in edge_payloads.items()
        },
        num_nodes={
            node_type: int(node_ids.numel())
            for node_type, node_ids in node_ids_by_type.items()
        },
    )
    return LocalGraphStoreAdapter(
        graph_store,
        boundary_edge_index_by_type={
            edge_type: boundary_edge_data["edge_index"]
            for edge_type, boundary_edge_data in _boundary_edge_data_by_type(payload, tuple(edge_payloads)).items()
        },
    )


def load_partitioned_stores(root) -> tuple[PartitionManifest, PartitionedFeatureStore, PartitionedGraphStore]:
    root = Path(root)
    manifest = load_partition_manifest(root / "manifest.json")
    cache = _PartitionStoreBundleCache(
        root,
        {partition.partition_id: partition for partition in manifest.partitions},
    )
    feature_stores = {}
    graph_stores = {}
    for partition in manifest.partitions:
        feature_stores[partition.partition_id] = _LazyPartitionFeatureStoreAdapter(cache, partition.partition_id)
        graph_stores[partition.partition_id] = _LazyPartitionGraphStoreAdapter(cache, partition)
    return manifest, PartitionedFeatureStore(feature_stores), PartitionedGraphStore(graph_stores)


__all__ = [
    "DistributedFeatureStore",
    "DistributedGraphStore",
    "LocalFeatureStoreAdapter",
    "LocalGraphStoreAdapter",
    "PartitionedFeatureStore",
    "PartitionedGraphStore",
    "load_partitioned_stores",
]
