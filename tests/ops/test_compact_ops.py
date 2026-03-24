import torch

from vgl import Graph
from vgl.ops import compact_nodes


def test_compact_nodes_relabels_graph_and_returns_mapping():
    graph = Graph.homo(
        edge_index=torch.tensor([[3, 5, 3], [5, 3, 7]]),
        x=torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]),
    )

    compacted, mapping = compact_nodes(graph, torch.tensor([3, 5, 7]))

    assert torch.equal(compacted.x, torch.tensor([[3.0], [5.0], [7.0]]))
    assert torch.equal(compacted.edge_index, torch.tensor([[0, 1, 0], [1, 0, 2]]))
    assert mapping == {3: 0, 5: 1, 7: 2}
