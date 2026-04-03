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


def test_compact_nodes_homo_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[3, 5, 3], [5, 3, 7]]),
        x=torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]),
    )

    def fail_tolist(self):
        raise AssertionError("homo compact_nodes should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    compacted, mapping = compact_nodes(graph, torch.tensor([3, 5, 7]))

    assert torch.equal(compacted.edge_index, torch.tensor([[0, 1, 0], [1, 0, 2]]))
    assert mapping == {3: 0, 5: 1, 7: 2}


def test_hetero_compact_nodes_relabels_relation_and_returns_per_type_mappings():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])},
            "institution": {"x": torch.tensor([[7.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1, 0], [2, 4, 0]]),
                "edge_weight": torch.tensor([1.0, 2.0, 3.0]),
            }
        },
    )

    compacted, mapping = compact_nodes(
        graph,
        {"author": torch.tensor([0, 1]), "paper": torch.tensor([0, 2, 4])},
        edge_type=("author", "writes", "paper"),
    )

    assert set(compacted.nodes) == {"author", "paper"}
    assert set(compacted.edges) == {("author", "writes", "paper")}
    assert torch.equal(compacted.nodes["author"].x, torch.tensor([[10.0], [20.0]]))
    assert torch.equal(compacted.nodes["paper"].x, torch.tensor([[1.0], [3.0], [5.0]]))
    assert torch.equal(compacted.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0, 1, 0], [1, 2, 0]]))
    assert torch.equal(compacted.edges[("author", "writes", "paper")].edge_weight, torch.tensor([1.0, 2.0, 3.0]))
    assert mapping == {"author": {0: 0, 1: 1}, "paper": {0: 0, 2: 1, 4: 2}}


def test_compact_nodes_hetero_avoids_tensor_tolist(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1, 0], [2, 4, 0]]),
                "edge_weight": torch.tensor([1.0, 2.0, 3.0]),
            }
        },
    )

    def fail_tolist(self):
        raise AssertionError("hetero compact_nodes should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    compacted, mapping = compact_nodes(
        graph,
        {"author": torch.tensor([0, 1]), "paper": torch.tensor([0, 2, 4])},
        edge_type=("author", "writes", "paper"),
    )

    assert torch.equal(compacted.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0, 1, 0], [1, 2, 0]]))
    assert mapping == {"author": {0: 0, 1: 1}, "paper": {0: 0, 2: 1, 4: 2}}
