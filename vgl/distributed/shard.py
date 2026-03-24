from dataclasses import dataclass, field
from pathlib import Path

import torch

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
        node_data = dict(graph_payload["node_data"])
        edge_data = dict(graph_payload.get("edge_data", {}))
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
            edges={EDGE_TYPE: graph_payload["edge_index"]},
            num_nodes={"node": int(node_ids.numel())},
        )
        schema = GraphSchema(
            node_types=("node",),
            edge_types=(EDGE_TYPE,),
            node_features={"node": tuple(node_data.keys())},
            edge_features={EDGE_TYPE: ("edge_index",) + tuple(edge_data.keys())},
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


__all__ = ["LocalGraphShard"]
