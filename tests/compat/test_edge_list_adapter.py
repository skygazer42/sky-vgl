import pytest
import torch

from vgl import Graph


def test_graph_round_trips_to_edge_list_tensor():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"edge_weight": torch.tensor([0.5, 1.5, 2.5])},
    )

    edge_list = graph.to_edge_list()
    restored = Graph.from_edge_list(
        edge_list,
        node_data={"x": graph.x},
        edge_data={"edge_weight": graph.edata["edge_weight"]},
    )

    assert torch.equal(edge_list, torch.tensor([[0, 1], [0, 2], [2, 2]]))
    assert torch.equal(restored.edge_index, graph.edge_index)
    assert torch.equal(restored.x, graph.x)
    assert torch.equal(restored.edata["edge_weight"], graph.edata["edge_weight"])


def test_from_edge_list_accepts_python_edge_pairs():
    graph = Graph.from_edge_list(
        [(0, 1), (1, 2), (1, 2)],
        node_data={"x": torch.tensor([[1.0], [2.0], [3.0]])},
        edge_data={"edge_weight": torch.tensor([0.5, 1.5, 2.5])},
    )

    assert torch.equal(graph.edge_index, torch.tensor([[0, 1, 1], [1, 2, 2]]))
    assert torch.equal(graph.x, torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.equal(graph.edata["edge_weight"], torch.tensor([0.5, 1.5, 2.5]))


def test_from_edge_list_avoids_tensor_item(monkeypatch):
    def fail_item(self):
        raise AssertionError("edge list import should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    graph = Graph.from_edge_list(torch.tensor([[0, 1], [1, 2]], dtype=torch.long))

    assert torch.equal(graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert graph.num_nodes() == 3


def test_from_edge_list_avoids_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("edge list import should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    graph = Graph.from_edge_list(torch.tensor([[0, 1], [1, 2]], dtype=torch.long))

    assert torch.equal(graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert graph.num_nodes() == 3


def test_graph_from_edge_list_preserves_explicit_num_nodes_for_isolates():
    graph = Graph.from_edge_list([(0, 1)], num_nodes=4)

    assert graph.num_nodes() == 4
    assert torch.equal(graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert graph.adjacency().shape == (4, 4)
    assert torch.equal(graph.to_edge_list(), torch.tensor([[0, 1]]))


def test_graph_from_edge_list_accepts_tensor_num_nodes_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("edge list import should stay off tensor.__int__ for num_nodes")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    graph = Graph.from_edge_list([(0, 1)], num_nodes=torch.tensor(4))

    assert graph.num_nodes() == 4
    assert torch.equal(graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert graph.adjacency().shape == (4, 4)


def test_graph_from_edge_list_accepts_transposed_tensor_input():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    graph = Graph.from_edge_list(edge_index)

    assert torch.equal(graph.edge_index, edge_index)
    assert torch.equal(graph.to_edge_list(), edge_index.t())


def test_to_edge_list_rejects_heterogeneous_graphs():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[3.0], [4.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [0, 1]]),
            }
        },
    )

    with pytest.raises(ValueError, match="homogeneous"):
        graph.to_edge_list()


def test_compat_exports_edge_list_helpers():
    from vgl.compat import from_edge_list, to_edge_list

    graph = Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.tensor([[1.0], [2.0]]))

    edge_list = to_edge_list(graph)
    restored = from_edge_list(edge_list, node_data={"x": graph.x})

    assert torch.equal(edge_list, torch.tensor([[0, 1]]))
    assert torch.equal(restored.edge_index, graph.edge_index)
    assert torch.equal(restored.x, graph.x)
