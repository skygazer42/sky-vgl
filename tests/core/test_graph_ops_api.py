import torch

from vgl import Graph


def test_graph_method_bridges_call_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    with_loops = graph.add_self_loops()
    subgraph = graph.node_subgraph(torch.tensor([0, 1, 2]))
    compacted, mapping = graph.compact_nodes(torch.tensor([0, 1, 2]))

    assert {tuple(edge) for edge in with_loops.edge_index.t().tolist()} == {(0, 1), (1, 2), (0, 0), (1, 1), (2, 2)}
    assert torch.equal(subgraph.edge_index, graph.edge_index)
    assert torch.equal(compacted.edge_index, graph.edge_index)
    assert mapping == {0: 0, 1: 1, 2: 2}
