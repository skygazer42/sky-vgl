from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PartitionShard:
    partition_id: int
    node_range: tuple[int, int]
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        partition_id = int(self.partition_id)
        start, end = self.node_range
        start = int(start)
        end = int(end)
        if partition_id < 0:
            raise ValueError("partition_id must be >= 0")
        if start < 0 or end < start:
            raise ValueError("partition node range must be valid")
        object.__setattr__(self, "partition_id", partition_id)
        object.__setattr__(self, "node_range", (start, end))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_nodes(self) -> int:
        return self.node_range[1] - self.node_range[0]

    def owns(self, node_id: int) -> bool:
        start, end = self.node_range
        return start <= int(node_id) < end

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "node_range": list(self.node_range),
            "path": self.path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    num_nodes: int
    partitions: tuple[PartitionShard, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        num_nodes = int(self.num_nodes)
        if num_nodes < 0:
            raise ValueError("num_nodes must be >= 0")
        partitions = tuple(self.partitions)
        partition_ids = [partition.partition_id for partition in partitions]
        if len(partition_ids) != len(set(partition_ids)):
            raise ValueError("partition ids must be unique")
        for partition in partitions:
            if partition.node_range[1] > num_nodes:
                raise ValueError("partition node range must fit within num_nodes")
        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_partitions(self) -> int:
        return len(self.partitions)

    def owner(self, node_id: int) -> PartitionShard:
        node_id = int(node_id)
        if node_id < 0 or node_id >= self.num_nodes:
            raise KeyError(node_id)
        for partition in self.partitions:
            if partition.owns(node_id):
                return partition
        raise KeyError(node_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "metadata": dict(self.metadata),
            "partitions": [partition.to_dict() for partition in self.partitions],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PartitionManifest":
        return cls(
            num_nodes=payload["num_nodes"],
            metadata=payload.get("metadata", {}),
            partitions=tuple(
                PartitionShard(
                    partition_id=partition["partition_id"],
                    node_range=tuple(partition["node_range"]),
                    path=partition.get("path"),
                    metadata=partition.get("metadata", {}),
                )
                for partition in payload.get("partitions", ())
            ),
        )


def save_partition_manifest(path, manifest: PartitionManifest) -> Path:
    path = Path(path)
    path.write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))
    return path


def load_partition_manifest(path) -> PartitionManifest:
    return PartitionManifest.from_dict(json.loads(Path(path).read_text()))


__all__ = [
    "PartitionManifest",
    "PartitionShard",
    "load_partition_manifest",
    "save_partition_manifest",
]
