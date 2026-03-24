import torch

from vgl import Graph
from vgl.ops import khop_nodes, khop_subgraph


def test_khop_nodes_expands_outbound_frontier():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 1]]),
        x=torch.randn(4, 2),
    )

    nodes = khop_nodes(graph, torch.tensor([0]), num_hops=2, direction="out")

    assert torch.equal(nodes, torch.tensor([0, 1, 2]))


def test_khop_nodes_expands_inbound_frontier():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.randn(4, 2),
    )

    nodes = khop_nodes(graph, torch.tensor([2]), num_hops=2, direction="in")

    assert torch.equal(nodes, torch.tensor([0, 1, 2, 3]))


def test_khop_subgraph_returns_relabelled_node_induced_subgraph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    subgraph = khop_subgraph(graph, torch.tensor([0]), num_hops=2)

    assert torch.equal(subgraph.x, torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1], [1, 2]]))
