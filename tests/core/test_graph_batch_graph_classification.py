import torch

from vgl import Graph
from vgl.core.batch import GraphBatch
from vgl.data.sample import SampleRecord


def test_graph_batch_tracks_graph_ptr_labels_and_metadata():
    graphs = [
        Graph.homo(
            edge_index=torch.tensor([[0], [1]]),
            x=torch.randn(2, 4),
            y=torch.tensor([1]),
        ),
        Graph.homo(
            edge_index=torch.tensor([[0], [1]]),
            x=torch.randn(3, 4),
            y=torch.tensor([0]),
        ),
    ]
    samples = [
        SampleRecord(graph=graphs[0], metadata={"label": 1}, sample_id="g0"),
        SampleRecord(graph=graphs[1], metadata={"label": 0}, sample_id="g1"),
    ]

    batch = GraphBatch.from_samples(samples, label_key="y", label_source="graph")

    assert batch.num_graphs == 2
    assert torch.equal(batch.graph_ptr, torch.tensor([0, 2, 5]))
    assert torch.equal(batch.labels, torch.tensor([1, 0]))
    assert batch.metadata[0]["label"] == 1

def test_graph_batch_tracks_typed_graph_ptrs_and_metadata_labels_for_hetero_samples():
    graphs = [
        Graph.hetero(
            nodes={
                "paper": {"x": torch.randn(2, 4)},
                "author": {"x": torch.randn(1, 4)},
            },
            edges={
                ("author", "writes", "paper"): {
                    "edge_index": torch.tensor([[0, 0], [0, 1]]),
                },
                ("paper", "written_by", "author"): {
                    "edge_index": torch.tensor([[0, 1], [0, 0]]),
                },
            },
        ),
        Graph.hetero(
            nodes={
                "paper": {"x": torch.randn(3, 4)},
                "author": {"x": torch.randn(2, 4)},
            },
            edges={
                ("author", "writes", "paper"): {
                    "edge_index": torch.tensor([[0, 1], [1, 2]]),
                },
                ("paper", "written_by", "author"): {
                    "edge_index": torch.tensor([[1, 2], [0, 1]]),
                },
            },
        ),
    ]
    samples = [
        SampleRecord(graph=graphs[0], metadata={"label": 1}, sample_id="h0"),
        SampleRecord(graph=graphs[1], metadata={"label": 0}, sample_id="h1"),
    ]

    batch = GraphBatch.from_samples(samples, label_key="label", label_source="metadata")

    assert batch.num_graphs == 2
    assert torch.equal(batch.labels, torch.tensor([1, 0]))
    assert batch.metadata[0]["label"] == 1
    assert torch.equal(batch.graph_ptr_by_type["paper"], torch.tensor([0, 2, 5]))
    assert torch.equal(batch.graph_ptr_by_type["author"], torch.tensor([0, 1, 3]))

