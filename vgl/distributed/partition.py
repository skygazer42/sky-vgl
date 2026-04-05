from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


def _normalize_range(node_range) -> tuple[int, int]:
    start, end = node_range
    start = int(start)
    end = int(end)
    if start < 0 or end < start:
        raise ValueError("partition node range must be valid")
    return start, end


_EDGE_ID_METADATA_KEYS = ("edge_ids_by_type", "boundary_edge_ids_by_type")


def _normalize_edge_type(edge_type) -> tuple[str, str, str]:
    if isinstance(edge_type, str):
        edge_type = json.loads(edge_type)
    if not isinstance(edge_type, (list, tuple)) or len(edge_type) != 3:
        raise ValueError("edge type metadata keys must be length-3 tuples")
    src_type, relation_type, dst_type = edge_type
    return str(src_type), str(relation_type), str(dst_type)


def _normalize_edge_id_metadata(raw_value) -> dict[tuple[str, str, str], tuple[int, ...]]:
    raw_value = dict(raw_value)
    return {
        _normalize_edge_type(edge_type): tuple(int(edge_id) for edge_id in edge_ids)
        for edge_type, edge_ids in raw_value.items()
    }


def _normalize_shape(shape) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _normalize_node_feature_shape_metadata(raw_value) -> dict[str, dict[str, tuple[int, ...]]]:
    raw_value = dict(raw_value)
    return {
        str(node_type): {
            str(feature_name): _normalize_shape(shape)
            for feature_name, shape in dict(feature_shapes).items()
        }
        for node_type, feature_shapes in raw_value.items()
    }


def _normalize_edge_feature_shape_metadata(raw_value) -> dict[tuple[str, str, str], dict[str, tuple[int, ...]]]:
    raw_value = dict(raw_value)
    return {
        _normalize_edge_type(edge_type): {
            str(feature_name): _normalize_shape(shape)
            for feature_name, shape in dict(feature_shapes).items()
        }
        for edge_type, feature_shapes in raw_value.items()
    }


def _normalize_partition_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    for key in _EDGE_ID_METADATA_KEYS:
        raw_value = normalized.get(key)
        if raw_value is None:
            continue
        normalized[key] = _normalize_edge_id_metadata(raw_value)
    raw_node_feature_shapes = normalized.get("node_feature_shapes")
    if raw_node_feature_shapes is not None:
        normalized["node_feature_shapes"] = _normalize_node_feature_shape_metadata(raw_node_feature_shapes)
    raw_edge_feature_shapes = normalized.get("edge_feature_shapes")
    if raw_edge_feature_shapes is not None:
        normalized["edge_feature_shapes"] = _normalize_edge_feature_shape_metadata(raw_edge_feature_shapes)
    return normalized


def _serialize_partition_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(metadata)
    for key in _EDGE_ID_METADATA_KEYS:
        raw_value = serialized.get(key)
        if raw_value is None:
            continue
        serialized[key] = {
            json.dumps(list(edge_type)): list(edge_ids)
            for edge_type, edge_ids in dict(raw_value).items()
        }
    raw_node_feature_shapes = serialized.get("node_feature_shapes")
    if raw_node_feature_shapes is not None:
        serialized["node_feature_shapes"] = {
            str(node_type): {
                str(feature_name): list(shape)
                for feature_name, shape in dict(feature_shapes).items()
            }
            for node_type, feature_shapes in dict(raw_node_feature_shapes).items()
        }
    raw_edge_feature_shapes = serialized.get("edge_feature_shapes")
    if raw_edge_feature_shapes is not None:
        serialized["edge_feature_shapes"] = {
            json.dumps(list(edge_type)): {
                str(feature_name): list(shape)
                for feature_name, shape in dict(feature_shapes).items()
            }
            for edge_type, feature_shapes in dict(raw_edge_feature_shapes).items()
        }
    return serialized


@dataclass(frozen=True, slots=True)
class PartitionShard:
    partition_id: int
    node_range: tuple[int, int]
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    node_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        partition_id = int(self.partition_id)
        if partition_id < 0:
            raise ValueError("partition_id must be >= 0")
        default_range = _normalize_range(self.node_range)
        raw_node_ranges = dict(self.node_ranges)
        if raw_node_ranges:
            node_ranges = {str(node_type): _normalize_range(node_range) for node_type, node_range in raw_node_ranges.items()}
        else:
            node_ranges = {"node": default_range}
        if "node" in node_ranges:
            default_range = node_ranges["node"]
        metadata = _normalize_partition_metadata(self.metadata)
        object.__setattr__(self, "partition_id", partition_id)
        object.__setattr__(self, "node_range", default_range)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "node_ranges", node_ranges)

    @property
    def num_nodes(self) -> int:
        return sum(end - start for start, end in self.node_ranges.values())

    def node_range_for(self, node_type: str = "node") -> tuple[int, int]:
        node_type = str(node_type)
        try:
            return self.node_ranges[node_type]
        except KeyError as exc:
            raise KeyError(node_type) from exc

    @property
    def edge_ids_by_type(self) -> dict[tuple[str, str, str], tuple[int, ...]]:
        return dict(self.metadata.get("edge_ids_by_type", {}))

    @property
    def boundary_edge_ids_by_type(self) -> dict[tuple[str, str, str], tuple[int, ...]]:
        return dict(self.metadata.get("boundary_edge_ids_by_type", {}))

    @property
    def node_feature_shapes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            str(node_type): dict(feature_shapes)
            for node_type, feature_shapes in self.metadata.get("node_feature_shapes", {}).items()
        }

    @property
    def edge_feature_shapes(self) -> dict[tuple[str, str, str], dict[str, tuple[int, ...]]]:
        return {
            _normalize_edge_type(edge_type): dict(feature_shapes)
            for edge_type, feature_shapes in self.metadata.get("edge_feature_shapes", {}).items()
        }

    def feature_shape(self, key) -> tuple[int, ...]:
        entity_kind, type_key, feature_name = key
        feature_name = str(feature_name)
        if entity_kind == "node":
            try:
                return self.node_feature_shapes[str(type_key)][feature_name]
            except KeyError as exc:
                raise KeyError(key) from exc
        if entity_kind == "edge":
            try:
                return self.edge_feature_shapes[tuple(type_key)][feature_name]
            except KeyError as exc:
                raise KeyError(key) from exc
        raise KeyError(key)

    def owns(self, node_id: int, *, node_type: str = "node") -> bool:
        start, end = self.node_range_for(node_type)
        return start <= int(node_id) < end

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "node_range": list(self.node_range),
            "node_ranges": {node_type: list(node_range) for node_type, node_range in self.node_ranges.items()},
            "path": self.path,
            "metadata": _serialize_partition_metadata(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    num_nodes: int
    partitions: tuple[PartitionShard, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    num_nodes_by_type: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        num_nodes = int(self.num_nodes)
        if num_nodes < 0:
            raise ValueError("num_nodes must be >= 0")
        partitions = tuple(self.partitions)
        partition_ids = [partition.partition_id for partition in partitions]
        if len(partition_ids) != len(set(partition_ids)):
            raise ValueError("partition ids must be unique")

        raw_num_nodes_by_type = dict(self.num_nodes_by_type)
        if not raw_num_nodes_by_type:
            raw_num_nodes_by_type = {"node": num_nodes}
        num_nodes_by_type = {str(node_type): int(count) for node_type, count in raw_num_nodes_by_type.items()}
        if any(count < 0 for count in num_nodes_by_type.values()):
            raise ValueError("typed node counts must be >= 0")
        if sum(num_nodes_by_type.values()) != num_nodes:
            raise ValueError("num_nodes must equal the sum of typed node counts")

        for partition in partitions:
            for node_type, (_, end) in partition.node_ranges.items():
                try:
                    typed_count = num_nodes_by_type[node_type]
                except KeyError as exc:
                    raise ValueError(f"partition node range references unknown node type {node_type!r}") from exc
                if end > typed_count:
                    raise ValueError("partition node range must fit within num_nodes")

        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "num_nodes_by_type", num_nodes_by_type)

    @property
    def num_partitions(self) -> int:
        return len(self.partitions)

    @property
    def time_attr(self) -> str | None:
        time_attr = self.metadata.get("time_attr")
        if time_attr is None:
            return None
        return str(time_attr)

    @property
    def edge_types(self) -> tuple[tuple[str, str, str], ...]:
        raw_edge_types = self.metadata.get("edge_types")
        if raw_edge_types is not None:
            return tuple(_normalize_edge_type(edge_type) for edge_type in raw_edge_types)

        ordered = []
        seen = set()
        for partition in self.partitions:
            for mapping in (partition.edge_ids_by_type, partition.boundary_edge_ids_by_type):
                for edge_type in mapping:
                    if edge_type in seen:
                        continue
                    seen.add(edge_type)
                    ordered.append(edge_type)
        return tuple(ordered)

    def owner(self, node_id: int, *, node_type: str = "node") -> PartitionShard:
        node_type = str(node_type)
        node_id = int(node_id)
        try:
            typed_count = self.num_nodes_by_type[node_type]
        except KeyError as exc:
            raise KeyError(node_type) from exc
        if node_id < 0 or node_id >= typed_count:
            raise KeyError(node_id)
        for partition in self.partitions:
            if partition.owns(node_id, node_type=node_type):
                return partition
        raise KeyError(node_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_nodes_by_type": dict(self.num_nodes_by_type),
            "metadata": dict(self.metadata),
            "partitions": [partition.to_dict() for partition in self.partitions],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PartitionManifest":
        return cls(
            num_nodes=payload["num_nodes"],
            num_nodes_by_type=payload.get("num_nodes_by_type", {}),
            metadata=payload.get("metadata", {}),
            partitions=tuple(
                PartitionShard(
                    partition_id=partition["partition_id"],
                    node_range=tuple(partition["node_range"]),
                    node_ranges={
                        node_type: tuple(node_range)
                        for node_type, node_range in partition.get("node_ranges", {}).items()
                    },
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
