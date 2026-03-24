from dataclasses import dataclass, field
from pathlib import Path

import torch

from vgl.data.ondisk import deserialize_graph
from vgl.distributed.partition import PartitionManifest, PartitionShard, load_partition_manifest
from vgl.graph.graph import Graph
from vgl.graph.schema import GraphSchema
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore


EDGE_TYPE = ("node", "to", "node")


@dataclass(slots=True)
class LocalGraphShard:
    manifest: PartitionManifest
    partition: PartitionShard
    root: Path
    node_ids: torch.Tensor
    feature_store: FeatureStore
    graph_store: InMemoryGraphStore
    graph: Graph
    _global_to_local: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._global_to_local = {int(node_id): index for index, node_id in enumerate(self.node_ids.tolist())}

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
        graph_payload = payload["graph"]
        graph = deserialize_graph(graph_payload)
        if set(graph.nodes) != {"node"} or len(graph.edges) != 1:
            raise ValueError("LocalGraphShard currently supports homogeneous partitions only")
        edge_type = graph._default_edge_type()
        node_data = dict(graph.nodes["node"].data)
        edge_store = graph.edges[edge_type]
        edge_data = {
            key: value
            for key, value in edge_store.data.items()
            if key != "edge_index"
        }
        node_ids = payload.get("node_ids", node_data.get("n_id"))
        if node_ids is None:
            start, end = partition.node_range
            node_ids = torch.arange(start, end, dtype=torch.long)
        node_ids = torch.as_tensor(node_ids, dtype=torch.long)

        feature_store = FeatureStore(
            {
                **{
                    ("node", "node", name): InMemoryTensorStore(value)
                    for name, value in node_data.items()
                    if isinstance(value, torch.Tensor)
                },
                **{
                    ("edge", EDGE_TYPE, name): InMemoryTensorStore(value)
                    for name, value in edge_data.items()
                    if isinstance(value, torch.Tensor)
                },
            }
        )
        graph_store = InMemoryGraphStore(
            edges={EDGE_TYPE: edge_store.edge_index},
            num_nodes={"node": int(node_ids.numel())},
        )
        schema = GraphSchema(
            node_types=("node",),
            edge_types=(EDGE_TYPE,),
            node_features={"node": tuple(node_data.keys())},
            edge_features={EDGE_TYPE: ("edge_index",) + tuple(edge_data.keys())},
            time_attr=graph.schema.time_attr,
        )
        graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)
        return cls(
            manifest=manifest,
            partition=partition,
            root=root,
            node_ids=node_ids,
            feature_store=feature_store,
            graph_store=graph_store,
            graph=graph,
        )

    def global_to_local(self, node_ids: torch.Tensor) -> torch.Tensor:
        values = []
        for node_id in torch.as_tensor(node_ids, dtype=torch.long).tolist():
            try:
                values.append(self._global_to_local[int(node_id)])
            except KeyError as exc:
                raise KeyError(f"node {node_id} is not present in partition {self.partition.partition_id}") from exc
        return torch.tensor(values, dtype=torch.long)

    def local_to_global(self, node_ids: torch.Tensor) -> torch.Tensor:
        local_ids = torch.as_tensor(node_ids, dtype=torch.long)
        if local_ids.numel() > 0 and ((local_ids < 0).any() or (local_ids >= self.node_ids.numel()).any()):
            raise IndexError(f"local node ids are out of range for partition {self.partition.partition_id}")
        return self.node_ids[local_ids]

    def global_edge_index(self) -> torch.Tensor:
        return self.local_to_global(self.graph_store.edge_index())


__all__ = ["LocalGraphShard"]
