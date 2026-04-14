import json
from pathlib import Path

import torch

from vgl._artifact import (
    ARTIFACT_FORMAT_KEY,
    ARTIFACT_FORMAT_VERSION_KEY,
    build_artifact_metadata,
    read_artifact_metadata,
)
from vgl.data.catalog import DatasetManifest, DatasetSplit
from vgl.graph.graph import Graph

GRAPH_PAYLOAD_FORMAT = "vgl.graph_payload"
GRAPH_PAYLOAD_FORMAT_VERSION = 1


def serialize_graph(graph: Graph) -> dict:
    return {
        **build_artifact_metadata(GRAPH_PAYLOAD_FORMAT, GRAPH_PAYLOAD_FORMAT_VERSION),
        "nodes": {
            node_type: dict(store.data)
            for node_type, store in graph.nodes.items()
        },
        "edges": {
            tuple(edge_type): dict(store.data)
            for edge_type, store in graph.edges.items()
        },
        "time_attr": graph.schema.time_attr,
    }


def deserialize_graph(payload: dict) -> Graph:
    if not isinstance(payload, dict):
        raise ValueError("graph payload must be a mapping")
    payload_format, _ = read_artifact_metadata(payload)
    if payload_format is not None and payload_format != GRAPH_PAYLOAD_FORMAT:
        raise ValueError(f"Unsupported graph payload format: {payload_format!r}")
    if "nodes" not in payload or "edges" not in payload:
        return Graph.homo(
            edge_index=payload["edge_index"],
            edge_data=dict(payload.get("edge_data", {})),
            **dict(payload["node_data"]),
        )

    nodes = {
        node_type: dict(node_data)
        for node_type, node_data in payload["nodes"].items()
    }
    edges = {
        tuple(edge_type): dict(edge_data)
        for edge_type, edge_data in payload["edges"].items()
    }
    time_attr = payload.get("time_attr")
    if time_attr is not None:
        return Graph.temporal(nodes=nodes, edges=edges, time_attr=time_attr)
    return Graph.hetero(nodes=nodes, edges=edges)


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


class OnDiskGraphDataset:
    def __init__(self, root, *, manifest=None, entries=None):
        self.root = Path(root)
        self.manifest = manifest_from_dict(json.loads((self.root / "manifest.json").read_text())) if manifest is None else manifest
        self._entries = tuple(self._load_entries() if entries is None else entries)

    def _load_entries(self):
        graphs_root = self.root / "graphs"
        if graphs_root.is_dir():
            graph_paths = tuple(sorted(graphs_root.glob("graph-*.pt")))
            legacy_path = self.root / "graphs.pt"
            if graph_paths or not legacy_path.exists():
                return graph_paths
        legacy_path = self.root / "graphs.pt"
        if legacy_path.exists():
            return tuple(torch.load(legacy_path, weights_only=True))
        return ()

    @staticmethod
    def _deserialize_entry(entry):
        if isinstance(entry, Path):
            return deserialize_graph(torch.load(entry, weights_only=True))
        return deserialize_graph(entry)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        return self._deserialize_entry(self._entries[index])

    def split(self, name):
        start = 0
        for split in self.manifest.splits:
            stop = start + split.size
            if split.name == name:
                if stop > len(self):
                    raise ValueError(f'manifest split {name!r} exceeds dataset size')
                return type(self)(self.root, manifest=self.manifest, entries=self._entries[start:stop])
            start = stop
        raise KeyError(name)

    @classmethod
    def write(cls, root, manifest: DatasetManifest, graphs) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        graphs_root = root / "graphs"
        graphs_root.mkdir(exist_ok=True)
        for existing in graphs_root.glob("graph-*.pt"):
            existing.unlink()
        (root / "manifest.json").write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))
        legacy_path = root / "graphs.pt"
        if legacy_path.exists():
            legacy_path.unlink()
        for index, graph in enumerate(graphs):
            graph_path = graphs_root / f"graph-{index:06d}.pt"
            torch.save(serialize_graph(graph), graph_path)
        return root


__all__ = ["OnDiskGraphDataset"]
