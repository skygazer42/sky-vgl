import json
from pathlib import Path

import torch

from vgl.data.catalog import DatasetManifest, DatasetSplit
from vgl.dataloading.dataset import ListDataset
from vgl.graph.graph import Graph


def serialize_graph(graph: Graph) -> dict:
    if set(graph.nodes) != {"node"} or len(graph.edges) != 1 or graph.schema.time_attr is not None:
        raise ValueError("OnDiskGraphDataset currently supports homogeneous non-temporal graphs only")
    edge_type = graph._default_edge_type()
    edge_store = graph.edges[edge_type]
    return {
        "edge_index": edge_store.edge_index,
        "node_data": dict(graph.nodes["node"].data),
        "edge_data": {
            key: value
            for key, value in edge_store.data.items()
            if key != "edge_index"
        },
    }


def deserialize_graph(payload: dict) -> Graph:
    return Graph.homo(
        edge_index=payload["edge_index"],
        edge_data=dict(payload.get("edge_data", {})),
        **dict(payload["node_data"]),
    )


def manifest_from_dict(payload: dict) -> DatasetManifest:
    return DatasetManifest(
        name=payload["name"],
        version=payload.get("version", "0"),
        description=payload.get("description"),
        metadata=payload.get("metadata", {}),
        splits=tuple(
            DatasetSplit(
                name=split["name"],
                size=split["size"],
                metadata=split.get("metadata", {}),
            )
            for split in payload.get("splits", ())
        ),
    )


class OnDiskGraphDataset(ListDataset):
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = manifest_from_dict(json.loads((self.root / "manifest.json").read_text()))
        payload = torch.load(self.root / "graphs.pt", weights_only=True)
        super().__init__([deserialize_graph(graph_payload) for graph_payload in payload])

    @classmethod
    def write(cls, root, manifest: DatasetManifest, graphs) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        graph_payload = [serialize_graph(graph) for graph in graphs]
        (root / "manifest.json").write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))
        torch.save(graph_payload, root / "graphs.pt")
        return root


__all__ = ["OnDiskGraphDataset"]
