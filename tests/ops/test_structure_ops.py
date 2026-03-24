import torch

from vgl import Graph
from vgl.ops import add_self_loops, remove_self_loops, to_bidirected


def _edge_pairs(graph):
    return {tuple(edge) for edge in graph.edge_index.t().tolist()}


def test_add_self_loops_adds_one_loop_per_node():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(3, 2),
    )

    updated = add_self_loops(graph)

    assert _edge_pairs(updated) == {(0, 1), (1, 0), (0, 0), (1, 1), (2, 2)}


def test_remove_self_loops_drops_existing_loops():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [0, 0, 1]]),
        x=torch.randn(2, 2),
    )

    updated = remove_self_loops(graph)

    assert _edge_pairs(updated) == {(1, 0)}


def test_to_bidirected_adds_missing_reverse_edges():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 2),
    )

    updated = to_bidirected(graph)

    assert _edge_pairs(updated) == {(0, 1), (1, 0), (1, 2), (2, 1)}
