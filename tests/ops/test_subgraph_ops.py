import torch

from vgl import Graph
from vgl.ops import edge_subgraph, node_subgraph


def test_node_subgraph_filters_edges_and_relabels_nodes():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    subgraph = node_subgraph(graph, torch.tensor([0, 2, 3]))

    assert torch.equal(subgraph.x, torch.tensor([[1.0], [3.0], [4.0]]))
    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))


def test_edge_subgraph_filters_edges_and_preserves_node_space():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"edge_weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    subgraph = edge_subgraph(graph, torch.tensor([1, 3]))

    assert torch.equal(subgraph.x, graph.x)
    assert torch.equal(subgraph.edge_index, torch.tensor([[2, 1], [3, 3]]))
    assert torch.equal(subgraph.edata["edge_weight"], torch.tensor([2.0, 4.0]))
