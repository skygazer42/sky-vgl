import torch

from vgl import Graph
from vgl.data.catalog import DatasetManifest, DatasetSplit
from vgl.data.ondisk import OnDiskGraphDataset


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
    assert (tmp_path / "graphs.pt").exists()
    assert dataset.manifest.name == "toy-graph"
    assert dataset.manifest.split("train").size == 2
    assert len(dataset) == 2
    assert torch.equal(dataset[0].edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(dataset[1].x, torch.tensor([[0.0], [1.0], [0.0]]))
