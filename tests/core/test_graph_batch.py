import torch

from vgl import Graph
from vgl.core.batch import GraphBatch


def test_graph_batch_tracks_membership():
    g1 = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    g2 = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([1, 0]),
    )

    batch = GraphBatch.from_graphs([g1, g2])

    assert batch.num_graphs == 2
    assert batch.graph_index.shape[0] == 4

def test_graph_batch_tracks_hetero_membership_per_node_type():
    g1 = Graph.hetero(
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
    )
    g2 = Graph.hetero(
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
    )

    batch = GraphBatch.from_graphs([g1, g2])

    assert batch.num_graphs == 2
    assert batch.graph_index is None
    assert batch.graph_ptr is None
    assert torch.equal(batch.graph_index_by_type["paper"], torch.tensor([0, 0, 1, 1, 1]))
    assert torch.equal(batch.graph_ptr_by_type["paper"], torch.tensor([0, 2, 5]))
    assert torch.equal(batch.graph_index_by_type["author"], torch.tensor([0, 1, 1]))
    assert torch.equal(batch.graph_ptr_by_type["author"], torch.tensor([0, 1, 3]))

