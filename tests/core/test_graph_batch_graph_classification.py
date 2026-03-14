import torch

from gnn import Graph
from gnn.core.batch import GraphBatch
from gnn.data.sample import SampleRecord


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
