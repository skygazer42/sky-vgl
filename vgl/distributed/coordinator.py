from dataclasses import dataclass
from typing import Protocol

import torch

from vgl.distributed.shard import LocalGraphShard
from vgl.distributed.store import DistributedFeatureStore, LocalFeatureStoreAdapter
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
    def route_node_ids(self, node_ids: torch.Tensor) -> tuple[ShardRoute, ...]:
        ...

    def fetch_node_features(self, key: FeatureKey, node_ids: torch.Tensor) -> TensorSlice:
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

    def route_node_ids(self, node_ids: torch.Tensor) -> tuple[ShardRoute, ...]:
        node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        grouped: dict[int, dict[str, list[int]]] = {}
        for position, node_id in enumerate(node_ids.tolist()):
            partition = self.manifest.owner(node_id)
            shard = self.shards.get(partition.partition_id)
            if shard is None:
                raise KeyError(f"missing shard for partition {partition.partition_id}")
            bucket = grouped.setdefault(
                partition.partition_id,
                {"global_ids": [], "local_ids": [], "positions": []},
            )
            bucket["global_ids"].append(int(node_id))
            bucket["positions"].append(int(position))
            bucket["local_ids"].append(int(shard.global_to_local(torch.tensor([node_id])).item()))

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
        node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        routes = self.route_node_ids(node_ids)
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


__all__ = ["LocalSamplingCoordinator", "SamplingCoordinator", "ShardRoute"]
