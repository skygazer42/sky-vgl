import networkx as nx
import pytest
import torch

from vgl import Graph


def test_graph_round_trips_to_networkx_multidigraph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 1, 2]]),
        x=torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        y=torch.tensor([0, 1, 0]),
        edge_data={
            "edge_weight": torch.tensor([0.5, 1.5, 2.5]),
            "e_id": torch.tensor([10, 11, 12]),
        },
    )

    nx_graph = graph.to_networkx()
    restored = Graph.from_networkx(nx_graph)

    assert isinstance(nx_graph, nx.MultiDiGraph)
    assert nx_graph.number_of_nodes() == 3
    assert nx_graph.number_of_edges() == 3
    assert torch.equal(nx_graph.nodes[0]["x"], torch.tensor([1.0, 0.0]))
    assert torch.equal(nx_graph.nodes[1]["y"], torch.tensor(1))
    assert torch.equal(nx_graph[0][1][0]["edge_weight"], torch.tensor(0.5))
    assert torch.equal(nx_graph[0][1][1]["e_id"], torch.tensor(11))
    assert torch.equal(restored.edge_index, graph.edge_index)
    assert torch.equal(restored.x, graph.x)
    assert torch.equal(restored.y, graph.y)
    assert torch.equal(restored.edata["edge_weight"], graph.edata["edge_weight"])
    assert torch.equal(restored.edata["e_id"], graph.edata["e_id"])


def test_to_networkx_uses_public_node_ids_as_exported_node_labels():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([[1.0], [2.0]]),
        n_id=torch.tensor([10, 20]),
    )

    nx_graph = graph.to_networkx()

    assert list(nx_graph.nodes()) == [10, 20]
    assert list(nx_graph.edges()) == [(10, 20)]
    assert torch.equal(nx_graph.nodes[10]["n_id"], torch.tensor(10))


def test_to_networkx_rejects_duplicate_public_node_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([[1.0], [2.0]]),
        n_id=torch.tensor([10, 10]),
    )

    with pytest.raises(ValueError, match="unique public node ids"):
        graph.to_networkx()


def test_to_networkx_rejects_non_integral_public_node_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([[1.0], [2.0]]),
        n_id=torch.tensor([1.5, 2.5]),
    )

    with pytest.raises(ValueError, match="integer public node ids"):
        graph.to_networkx()


def test_from_networkx_imports_external_directed_graph_with_tensor_attributes():
    nx_graph = nx.DiGraph()
    nx_graph.add_node(0, x=torch.tensor([1.0, 0.0]), y=torch.tensor(1))
    nx_graph.add_node(1, x=torch.tensor([2.0, 0.0]), y=torch.tensor(0))
    nx_graph.add_node(2, x=torch.tensor([3.0, 0.0]), y=torch.tensor(1))
    nx_graph.add_edge(0, 1, edge_weight=torch.tensor([0.5, 0.6]))
    nx_graph.add_edge(1, 2, edge_weight=torch.tensor([1.5, 1.6]))

    graph = Graph.from_networkx(nx_graph)

    assert torch.equal(graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(graph.x, torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert torch.equal(graph.y, torch.tensor([1, 0, 1]))
    assert torch.equal(graph.edata["edge_weight"], torch.tensor([[0.5, 0.6], [1.5, 1.6]]))


def test_graph_from_networkx_preserves_parallel_edges_from_multidigraph():
    nx_graph = nx.MultiDiGraph()
    nx_graph.add_node(0, x=torch.tensor([1.0]))
    nx_graph.add_node(1, x=torch.tensor([2.0]))
    nx_graph.add_node(2, x=torch.tensor([3.0]))
    nx_graph.add_edge(0, 1, edge_weight=torch.tensor(0.5))
    nx_graph.add_edge(0, 1, edge_weight=torch.tensor(1.5))
    nx_graph.add_edge(1, 2, edge_weight=torch.tensor(2.5))

    graph = Graph.from_networkx(nx_graph)
    restored = graph.to_networkx()

    assert torch.equal(graph.edge_index, torch.tensor([[0, 0, 1], [1, 1, 2]]))
    assert torch.equal(graph.edata["edge_weight"], torch.tensor([0.5, 1.5, 2.5]))
    assert restored.number_of_edges() == 3
    assert torch.equal(restored[0][1][0]["edge_weight"], torch.tensor(0.5))
    assert torch.equal(restored[0][1][1]["edge_weight"], torch.tensor(1.5))


def test_from_networkx_preserves_numeric_multigraph_edge_keys_as_public_edge_ids():
    nx_graph = nx.MultiDiGraph()
    nx_graph.add_node(0, x=torch.tensor([1.0]))
    nx_graph.add_node(1, x=torch.tensor([2.0]))
    nx_graph.add_edge(0, 1, key=10, edge_weight=torch.tensor(0.5))
    nx_graph.add_edge(0, 1, key=11, edge_weight=torch.tensor(1.5))

    graph = Graph.from_networkx(nx_graph)

    assert torch.equal(graph.edge_index, torch.tensor([[0, 0], [1, 1]]))
    assert torch.equal(graph.edata["edge_weight"], torch.tensor([0.5, 1.5]))
    assert torch.equal(graph.edata["e_id"], torch.tensor([10, 11]))


def test_from_networkx_preserves_numeric_external_node_ids():
    nx_graph = nx.DiGraph()
    nx_graph.add_node(10, x=torch.tensor([1.0]))
    nx_graph.add_node(20, x=torch.tensor([2.0]))
    nx_graph.add_edge(10, 20, edge_weight=torch.tensor(0.5))

    graph = Graph.from_networkx(nx_graph)

    assert torch.equal(graph.n_id, torch.tensor([10, 20]))
    assert torch.equal(graph.edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(graph.edata["edge_weight"], torch.tensor([0.5]))


def test_from_networkx_preserves_num_nodes_for_featureless_graphs():
    nx_graph = nx.MultiDiGraph()
    nx_graph.add_node("a")
    nx_graph.add_node("b")
    nx_graph.add_node("c")
    nx_graph.add_edge("a", "b")

    graph = Graph.from_networkx(nx_graph)

    assert graph._node_count("node") == 3
    assert torch.equal(graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(graph.edge_index, torch.tensor([[0], [1]]))


def test_from_networkx_preserves_numeric_external_node_ids_for_featureless_graphs():
    nx_graph = nx.MultiDiGraph()
    nx_graph.add_node(10)
    nx_graph.add_node(20)
    nx_graph.add_node(30)
    nx_graph.add_edge(10, 20)

    graph = Graph.from_networkx(nx_graph)

    assert graph._node_count("node") == 3
    assert torch.equal(graph.n_id, torch.tensor([10, 20, 30]))
    assert torch.equal(graph.edge_index, torch.tensor([[0], [1]]))


def test_from_networkx_does_not_truncate_non_integral_numeric_node_labels():
    nx_graph = nx.DiGraph()
    nx_graph.add_node(1.5, x=torch.tensor([1.0]))
    nx_graph.add_node(2.5, x=torch.tensor([2.0]))
    nx_graph.add_edge(1.5, 2.5)

    graph = Graph.from_networkx(nx_graph)

    assert "n_id" not in graph.nodes["node"].data
    assert torch.equal(graph.edge_index, torch.tensor([[0], [1]]))


def test_from_networkx_preserves_num_nodes_for_featureless_edgeless_graphs():
    nx_graph = nx.MultiDiGraph()
    nx_graph.add_node("isolated")

    graph = Graph.from_networkx(nx_graph)

    assert graph._node_count("node") == 1
    assert torch.equal(graph.n_id, torch.tensor([0]))
    assert torch.equal(graph.edge_index, torch.empty((2, 0), dtype=torch.long))


def test_to_networkx_avoids_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 1, 2]]),
        x=torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        y=torch.tensor([0, 1, 0]),
        edge_data={
            "edge_weight": torch.tensor([0.5, 1.5, 2.5]),
            "e_id": torch.tensor([10, 11, 12]),
        },
    )

    def fail_int(self):
        raise AssertionError("NetworkX export should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    nx_graph = graph.to_networkx()

    assert isinstance(nx_graph, nx.MultiDiGraph)
    assert nx_graph.number_of_edges() == 3
    assert torch.equal(nx_graph[0][1][0]["edge_weight"], torch.tensor(0.5))


def test_to_networkx_rejects_heterogeneous_graphs():
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
        graph.to_networkx()


def test_from_networkx_rejects_undirected_graphs():
    nx_graph = nx.Graph()
    nx_graph.add_node(0, x=torch.tensor([1.0]))
    nx_graph.add_node(1, x=torch.tensor([2.0]))
    nx_graph.add_edge(0, 1, edge_weight=torch.tensor(0.5))

    with pytest.raises(ValueError, match="directed"):
        Graph.from_networkx(nx_graph)


def test_compat_exports_networkx_helpers():
    from vgl.compat import from_networkx, to_networkx

    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([[1.0], [2.0]]),
    )

    nx_graph = to_networkx(graph)
    restored = from_networkx(nx_graph)

    assert isinstance(nx_graph, nx.MultiDiGraph)
    assert torch.equal(restored.edge_index, graph.edge_index)
    assert torch.equal(restored.x, graph.x)
