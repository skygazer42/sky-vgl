from dataclasses import dataclass, field
from pathlib import Path

import torch

from vgl.data.ondisk import deserialize_graph
from vgl.distributed.partition import PartitionManifest, PartitionShard, load_partition_manifest
from vgl.graph.graph import Graph
from vgl.graph.schema import GraphSchema
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore


@dataclass(slots=True)
class LocalGraphShard:
    manifest: PartitionManifest
    partition: PartitionShard
    root: Path
    node_ids_by_type: dict[str, torch.Tensor]
    feature_store: FeatureStore
    graph_store: InMemoryGraphStore
    graph: Graph
    _global_to_local_by_type: dict[str, dict[int, int]] = field(default_factory=dict, repr=False)
    _global_to_local_edge_by_type: dict[tuple[str, str, str], dict[int, int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._global_to_local_by_type = {
            node_type: {int(node_id): index for index, node_id in enumerate(node_ids.tolist())}
            for node_type, node_ids in self.node_ids_by_type.items()
        }
        self._global_to_local_edge_by_type = {
            edge_type: {
                int(edge_id): index
                for index, edge_id in enumerate(self.edge_ids(edge_type=edge_type).tolist())
            }
            for edge_type in self.graph.edges
        }

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

        payload = torch.load(root / partition.path, weights_only=True)
        graph = deserialize_graph(payload["graph"])
        node_data_by_type = {node_type: dict(store.data) for node_type, store in graph.nodes.items()}
        edge_types = tuple(graph.edges)
        edge_feature_data = {
            edge_type: {
                key: value
                for key, value in edge_store.data.items()
                if key != "edge_index"
            }
            for edge_type, edge_store in graph.edges.items()
        }

        raw_node_ids = payload.get("node_ids")
        node_ids_by_type = {}
        if isinstance(raw_node_ids, dict):
            node_ids_by_type = {
                str(node_type): torch.as_tensor(node_ids, dtype=torch.long)
                for node_type, node_ids in raw_node_ids.items()
            }
        elif raw_node_ids is not None and len(graph.nodes) == 1:
            only_node_type = next(iter(graph.nodes))
            node_ids_by_type[only_node_type] = torch.as_tensor(raw_node_ids, dtype=torch.long)

        for node_type, node_data in node_data_by_type.items():
            if node_type in node_ids_by_type:
                continue
            node_ids = node_data.get("n_id")
            if node_ids is None:
                start, end = partition.node_range_for(node_type)
                node_ids = torch.arange(start, end, dtype=torch.long)
            node_ids_by_type[node_type] = torch.as_tensor(node_ids, dtype=torch.long)

        feature_store = FeatureStore(
            {
                **{
                    ("node", node_type, name): InMemoryTensorStore(value)
                    for node_type, node_data in node_data_by_type.items()
                    for name, value in node_data.items()
                    if isinstance(value, torch.Tensor)
                },
                **{
                    ("edge", edge_type, name): InMemoryTensorStore(value)
                    for edge_type, edge_data in edge_feature_data.items()
                    for name, value in edge_data.items()
                    if isinstance(value, torch.Tensor)
                },
            }
        )
        graph_store = InMemoryGraphStore(
            edges={edge_type: graph.edges[edge_type].edge_index for edge_type in edge_types},
            num_nodes={node_type: int(node_ids.numel()) for node_type, node_ids in node_ids_by_type.items()},
        )
        schema = GraphSchema(
            node_types=graph.schema.node_types,
            edge_types=edge_types,
            node_features={node_type: tuple(node_data.keys()) for node_type, node_data in node_data_by_type.items()},
            edge_features={
                edge_type: ("edge_index",) + tuple(edge_feature_data[edge_type].keys())
                for edge_type in edge_types
            },
            time_attr=graph.schema.time_attr,
        )
        graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)
        return cls(
            manifest=manifest,
            partition=partition,
            root=root,
            node_ids_by_type=node_ids_by_type,
            feature_store=feature_store,
            graph_store=graph_store,
            graph=graph,
        )

    def global_to_local(self, node_ids: torch.Tensor, *, node_type: str = "node") -> torch.Tensor:
        node_type = str(node_type)
        try:
            index = self._global_to_local_by_type[node_type]
        except KeyError as exc:
            raise KeyError(node_type) from exc
        values = []
        for node_id in torch.as_tensor(node_ids, dtype=torch.long).tolist():
            try:
                values.append(index[int(node_id)])
            except KeyError as exc:
                raise KeyError(
                    f"node {node_id} is not present in partition {self.partition.partition_id} for node type {node_type!r}"
                ) from exc
        return torch.tensor(values, dtype=torch.long)

    def local_to_global(self, node_ids: torch.Tensor, *, node_type: str = "node") -> torch.Tensor:
        global_ids = self.node_ids_for(node_type)
        local_ids = torch.as_tensor(node_ids, dtype=torch.long)
        if local_ids.numel() > 0 and ((local_ids < 0).any() or (local_ids >= global_ids.numel()).any()):
            raise IndexError(
                f"local node ids are out of range for partition {self.partition.partition_id} and node type {node_type!r}"
            )
        return global_ids[local_ids]

    def edge_ids(self, *, edge_type=None) -> torch.Tensor:
        if edge_type is None:
            edge_type = self.graph._default_edge_type()
        edge_type = tuple(edge_type)
        return torch.as_tensor(self.graph.edges[edge_type].data["e_id"], dtype=torch.long)

    def global_to_local_edge(self, edge_ids: torch.Tensor, *, edge_type=None) -> torch.Tensor:
        if edge_type is None:
            edge_type = self.graph._default_edge_type()
        edge_type = tuple(edge_type)
        try:
            index = self._global_to_local_edge_by_type[edge_type]
        except KeyError as exc:
            raise KeyError(edge_type) from exc
        values = []
        for edge_id in torch.as_tensor(edge_ids, dtype=torch.long).tolist():
            try:
                values.append(index[int(edge_id)])
            except KeyError as exc:
                raise KeyError(
                    f"edge {edge_id} is not present in partition {self.partition.partition_id} for edge type {edge_type!r}"
                ) from exc
        return torch.tensor(values, dtype=torch.long)

    def local_to_global_edge(self, edge_ids: torch.Tensor, *, edge_type=None) -> torch.Tensor:
        if edge_type is None:
            edge_type = self.graph._default_edge_type()
        edge_type = tuple(edge_type)
        global_ids = self.edge_ids(edge_type=edge_type)
        local_ids = torch.as_tensor(edge_ids, dtype=torch.long)
        if local_ids.numel() > 0 and ((local_ids < 0).any() or (local_ids >= global_ids.numel()).any()):
            raise IndexError(
                f"local edge ids are out of range for partition {self.partition.partition_id} and edge type {edge_type!r}"
            )
        return global_ids[local_ids]

    def global_edge_index(self, *, edge_type=None) -> torch.Tensor:
        if edge_type is None:
            edge_type = self.graph._default_edge_type()
        edge_type = tuple(edge_type)
        src_type, _, dst_type = edge_type
        local_edge_index = self.graph_store.edge_index(edge_type)
        return torch.stack(
            (
                self.local_to_global(local_edge_index[0], node_type=src_type),
                self.local_to_global(local_edge_index[1], node_type=dst_type),
            )
        )


__all__ = ["LocalGraphShard"]
