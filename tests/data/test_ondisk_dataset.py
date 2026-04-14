import json

import torch

from vgl._artifact import ARTIFACT_FORMAT_KEY, ARTIFACT_FORMAT_VERSION_KEY
from vgl import Graph
from vgl.data.catalog import DatasetManifest, DatasetSplit
from vgl.data.ondisk import (
    GRAPH_PAYLOAD_FORMAT,
    GRAPH_PAYLOAD_FORMAT_VERSION,
    OnDiskGraphDataset,
    deserialize_graph,
    serialize_graph,
)


def _graphs():
    return [
        Graph.homo(
            edge_index=torch.tensor([[0], [1]]),
            x=torch.tensor([[1.0], [0.0]]),
            y=torch.tensor([1]),
        ),
        Graph.homo(
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            x=torch.tensor([[0.0], [1.0], [0.0]]),
            y=torch.tensor([0]),
        ),
    ]


def test_ondisk_graph_dataset_round_trips_manifest_and_graphs(tmp_path):
    manifest = DatasetManifest(
        name="toy-graph",
        version="1.0",
        splits=(DatasetSplit("train", size=2),),
        metadata={"source": "fixture"},
    )

    OnDiskGraphDataset.write(tmp_path, manifest, _graphs())
    dataset = OnDiskGraphDataset(tmp_path)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "graphs").is_dir()
    assert (tmp_path / "graphs" / "graph-000000.pt").exists()
    assert (tmp_path / "graphs" / "graph-000001.pt").exists()
    assert dataset.manifest.name == "toy-graph"
    assert dataset.manifest.split("train").size == 2
    assert len(dataset) == 2
    assert torch.equal(dataset[0].edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(dataset[1].x, torch.tensor([[0.0], [1.0], [0.0]]))


def test_serialize_graph_emits_explicit_format_metadata():
    payload = serialize_graph(_graphs()[0])

    assert payload[ARTIFACT_FORMAT_KEY] == GRAPH_PAYLOAD_FORMAT
    assert payload[ARTIFACT_FORMAT_VERSION_KEY] == GRAPH_PAYLOAD_FORMAT_VERSION
    assert "nodes" in payload
    assert "edges" in payload


def test_deserialize_graph_accepts_legacy_payload_without_format_metadata():
    legacy_payload = {
        "nodes": {"node": {"x": torch.tensor([[1.0], [0.0]])}},
        "edges": {
            ("node", "to", "node"): {
                "edge_index": torch.tensor([[0], [1]]),
                "weight": torch.tensor([1.0]),
            }
        },
        "time_attr": None,
    }

    restored = deserialize_graph(legacy_payload)

    assert torch.equal(restored.edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(restored.edata["weight"], torch.tensor([1.0]))


def test_ondisk_graph_dataset_round_trips_heterogeneous_graphs(tmp_path):
    manifest = DatasetManifest(
        name="toy-hetero",
        version="1.0",
        splits=(DatasetSplit("train", size=1),),
    )
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[3.0], [4.0], [5.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "weight": torch.tensor([0.5, 1.5]),
            }
        },
    )

    OnDiskGraphDataset.write(tmp_path, manifest, [graph])
    dataset = OnDiskGraphDataset(tmp_path)
    restored = dataset[0]

    assert restored.schema.node_types == ("author", "paper")
    assert restored.schema.edge_types == (("author", "writes", "paper"),)
    assert torch.equal(restored.nodes["author"].x, torch.tensor([[1.0], [2.0]]))
    assert torch.equal(
        restored.edges[("author", "writes", "paper")].edge_index,
        torch.tensor([[0, 1], [1, 2]]),
    )
    assert torch.equal(
        restored.edges[("author", "writes", "paper")].weight,
        torch.tensor([0.5, 1.5]),
    )


def test_ondisk_graph_dataset_round_trips_temporal_graphs(tmp_path):
    manifest = DatasetManifest(
        name="toy-temporal",
        version="1.0",
        splits=(DatasetSplit("train", size=1),),
    )
    graph = Graph.temporal(
        nodes={"node": {"x": torch.tensor([[1.0], [2.0]])}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "timestamp": torch.tensor([3, 5]),
            }
        },
        time_attr="timestamp",
    )

    OnDiskGraphDataset.write(tmp_path, manifest, [graph])
    dataset = OnDiskGraphDataset(tmp_path)
    restored = dataset[0]

    assert restored.schema.time_attr == "timestamp"
    assert torch.equal(restored.edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(restored.edata["timestamp"], torch.tensor([3, 5]))


def test_ondisk_graph_dataset_writes_lazy_graph_layout(tmp_path):
    manifest = DatasetManifest(
        name="toy-graph",
        version="1.0",
        splits=(DatasetSplit("train", size=2),),
    )

    OnDiskGraphDataset.write(tmp_path, manifest, _graphs())
    dataset = OnDiskGraphDataset(tmp_path)

    assert (tmp_path / "graphs").is_dir()
    assert (tmp_path / "graphs" / "graph-000000.pt").exists()
    assert (tmp_path / "graphs" / "graph-000001.pt").exists()
    assert torch.equal(dataset[0].edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(dataset[1].x, torch.tensor([[0.0], [1.0], [0.0]]))


def test_ondisk_graph_dataset_exposes_contiguous_split_views(tmp_path):
    manifest = DatasetManifest(
        name="toy-splits",
        version="1.0",
        splits=(
            DatasetSplit("train", size=1),
            DatasetSplit("test", size=1),
        ),
    )

    OnDiskGraphDataset.write(tmp_path, manifest, _graphs())
    dataset = OnDiskGraphDataset(tmp_path)

    train = dataset.split("train")
    test = dataset.split("test")

    assert len(train) == 1
    assert len(test) == 1
    assert torch.equal(train[0].x, torch.tensor([[1.0], [0.0]]))
    assert torch.equal(test[0].x, torch.tensor([[0.0], [1.0], [0.0]]))


def test_ondisk_graph_dataset_loads_legacy_graphs_file(tmp_path):
    manifest = DatasetManifest(
        name="toy-legacy",
        version="1.0",
        splits=(DatasetSplit("train", size=2),),
    )

    (tmp_path / "manifest.json").write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))
    torch.save([serialize_graph(graph) for graph in _graphs()], tmp_path / "graphs.pt")

    dataset = OnDiskGraphDataset(tmp_path)

    assert len(dataset) == 2
    assert torch.equal(dataset[0].edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(dataset[1].x, torch.tensor([[0.0], [1.0], [0.0]]))
