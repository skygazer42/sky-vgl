import torch

from vgl import Graph
from vgl.ops import add_self_loops, remove_self_loops, to_bidirected, to_simple


def _edge_pairs(graph):
    return {tuple(edge) for edge in graph.edge_index.t().tolist()}


def test_add_self_loops_adds_one_loop_per_node():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(3, 2),
    )

    updated = add_self_loops(graph)

    assert _edge_pairs(updated) == {(0, 1), (1, 0), (0, 0), (1, 1), (2, 2)}


def test_add_self_loops_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(3, 2),
    )

    def fail_tolist(self):
        raise AssertionError("add_self_loops should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    updated = add_self_loops(graph)

    assert torch.equal(updated.edge_index, torch.tensor([[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]]))


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


def test_to_bidirected_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 2),
    )

    def fail_tolist(self):
        raise AssertionError("to_bidirected should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    updated = to_bidirected(graph)

    assert torch.equal(updated.edge_index, torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]]))


def test_to_bidirected_avoids_torch_isin(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 2),
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("to_bidirected should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    updated = to_bidirected(graph)

    assert torch.equal(updated.edge_index, torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]]))


def test_to_simple_collapses_parallel_edges_in_stable_order_and_tracks_counts():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 0], [1, 1, 2, 1]]),
        x=torch.randn(3, 2),
        edge_data={
            "weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "e_id": torch.tensor([10, 11, 12, 13]),
        },
    )

    simplified = to_simple(graph, count_attr="count")

    assert torch.equal(simplified.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(simplified.edata["weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(simplified.edata["count"], torch.tensor([3, 1]))
    assert "e_id" not in simplified.edata


def test_to_simple_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 0], [1, 1, 2, 1]]),
        x=torch.randn(3, 2),
        edge_data={
            "weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "e_id": torch.tensor([10, 11, 12, 13]),
        },
    )

    def fail_tolist(self):
        raise AssertionError("to_simple should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    simplified = to_simple(graph, count_attr="count")

    assert torch.equal(simplified.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(simplified.edata["weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(simplified.edata["count"], torch.tensor([3, 1]))


def test_to_simple_avoids_torch_unique(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 0], [1, 1, 2, 1]]),
        x=torch.randn(3, 2),
        edge_data={
            "weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "e_id": torch.tensor([10, 11, 12, 13]),
        },
    )

    def fail_unique(*args, **kwargs):
        raise AssertionError("to_simple should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    simplified = to_simple(graph, count_attr="count")

    assert torch.equal(simplified.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(simplified.edata["weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(simplified.edata["count"], torch.tensor([3, 1]))


def test_to_simple_updates_only_selected_heterogeneous_relation():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 0, 1], [1, 1, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0]),
                "e_id": torch.tensor([20, 21, 22]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 1], [2, 0]]),
                "score": torch.tensor([5.0, 6.0]),
            },
        },
    )

    simplified = to_simple(graph, edge_type=writes, count_attr="multiplicity")

    assert set(simplified.edges) == {writes, cites}
    assert torch.equal(simplified.edges[writes].edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(simplified.edges[writes].data["weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(simplified.edges[writes].data["multiplicity"], torch.tensor([2, 1]))
    assert "e_id" not in simplified.edges[writes].data
    assert torch.equal(simplified.edges[cites].edge_index, graph.edges[cites].edge_index)
    assert torch.equal(simplified.edges[cites].data["score"], graph.edges[cites].data["score"])
