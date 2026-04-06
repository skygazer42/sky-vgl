import pytest
import torch

from vgl import Graph
from vgl.ops import khop_nodes, khop_subgraph
from vgl.ops.khop import expand_neighbors


def _hetero_graph():
    return Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
            "institution": {"x": torch.tensor([[7.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "edge_weight": torch.tensor([1.0, 2.0, 3.0]),
            }
        },
    )


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


def test_khop_nodes_directional_expansion_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 1]]),
        x=torch.randn(4, 2),
    )

    def fail_tolist(self):
        raise AssertionError("homogeneous khop directional expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    nodes = khop_nodes(graph, torch.tensor([0]), num_hops=2, direction="out")

    assert torch.equal(nodes, torch.tensor([0, 1, 2]))


def test_expand_neighbors_homo_avoids_torch_isin(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.randn(4, 2),
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("homogeneous neighbor expansion should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    nodes = expand_neighbors(graph, torch.tensor([1]), num_neighbors=[-1])

    assert torch.equal(nodes, torch.tensor([0, 1, 2, 3]))


def test_expand_neighbors_homo_avoids_repeat_interleave(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.randn(4, 2),
    )

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("homogeneous neighbor expansion should avoid repeat_interleave interval expansion")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    nodes = expand_neighbors(graph, torch.tensor([1]), num_neighbors=[-1])

    assert torch.equal(nodes, torch.tensor([0, 1, 2, 3]))


def test_expand_neighbors_homo_avoids_tensor_int_for_fanouts(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.randn(4, 2),
    )

    def fail_int(self):
        raise AssertionError("neighbor fanout normalization should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    nodes = expand_neighbors(graph, torch.tensor([1]), num_neighbors=[torch.tensor(-1)])

    assert torch.equal(nodes, torch.tensor([0, 1, 2, 3]))


def test_khop_nodes_directional_expansion_avoids_torch_isin(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 1]]),
        x=torch.randn(4, 2),
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("homogeneous khop directional expansion should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    nodes = khop_nodes(graph, torch.tensor([0]), num_hops=2, direction="out")

    assert torch.equal(nodes, torch.tensor([0, 1, 2]))


def test_khop_nodes_directional_expansion_avoids_torch_unique(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 1]]),
        x=torch.randn(4, 2),
    )

    def fail_unique(*args, **kwargs):
        raise AssertionError("homogeneous khop directional expansion should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    nodes = khop_nodes(graph, torch.tensor([0]), num_hops=2, direction="out")

    assert torch.equal(nodes, torch.tensor([0, 1, 2]))


def test_khop_subgraph_returns_relabelled_node_induced_subgraph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    subgraph = khop_subgraph(graph, torch.tensor([0]), num_hops=2)

    assert torch.equal(subgraph.x, torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1], [1, 2]]))


def test_hetero_khop_nodes_expands_relation_local_outbound_frontier():
    graph = _hetero_graph()

    nodes = khop_nodes(
        graph,
        {"author": torch.tensor([1])},
        num_hops=2,
        direction="out",
        edge_type=("author", "writes", "paper"),
    )

    assert set(nodes) == {"author", "paper"}
    assert torch.equal(nodes["author"], torch.tensor([1]))
    assert torch.equal(nodes["paper"], torch.tensor([0, 2]))


def test_hetero_khop_nodes_expands_relation_local_inbound_frontier():
    graph = _hetero_graph()

    nodes = khop_nodes(
        graph,
        {"paper": torch.tensor([2])},
        num_hops=2,
        direction="in",
        edge_type=("author", "writes", "paper"),
    )

    assert set(nodes) == {"author", "paper"}
    assert torch.equal(nodes["author"], torch.tensor([1]))
    assert torch.equal(nodes["paper"], torch.tensor([2]))


def test_hetero_khop_nodes_directional_expansion_avoids_tensor_tolist(monkeypatch):
    graph = _hetero_graph()

    def fail_tolist(self):
        raise AssertionError("heterogeneous khop directional expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    nodes = khop_nodes(
        graph,
        {"author": torch.tensor([1])},
        num_hops=2,
        direction="out",
        edge_type=("author", "writes", "paper"),
    )

    assert set(nodes) == {"author", "paper"}
    assert torch.equal(nodes["author"], torch.tensor([1]))
    assert torch.equal(nodes["paper"], torch.tensor([0, 2]))


def test_expand_neighbors_hetero_avoids_torch_isin(monkeypatch):
    graph = _hetero_graph()

    def fail_isin(*args, **kwargs):
        raise AssertionError("heterogeneous neighbor expansion should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    nodes = expand_neighbors(
        graph,
        torch.tensor([1]),
        num_neighbors=[-1],
        node_type="paper",
    )

    assert torch.equal(nodes["author"], torch.tensor([0]))
    assert torch.equal(nodes["paper"], torch.tensor([1]))


def test_expand_neighbors_hetero_avoids_torch_unique(monkeypatch):
    graph = _hetero_graph()

    def fail_unique(*args, **kwargs):
        raise AssertionError("heterogeneous neighbor expansion should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    nodes = expand_neighbors(
        graph,
        torch.tensor([1]),
        num_neighbors=[-1],
        node_type="paper",
    )

    assert torch.equal(nodes["author"], torch.tensor([0]))
    assert torch.equal(nodes["paper"], torch.tensor([1]))


def test_hetero_khop_nodes_directional_expansion_avoids_repeat_interleave(monkeypatch):
    graph = _hetero_graph()

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("heterogeneous khop directional expansion should avoid repeat_interleave interval expansion")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    nodes = khop_nodes(
        graph,
        {"author": torch.tensor([1])},
        num_hops=2,
        direction="out",
        edge_type=("author", "writes", "paper"),
    )

    assert set(nodes) == {"author", "paper"}
    assert torch.equal(nodes["author"], torch.tensor([1]))
    assert torch.equal(nodes["paper"], torch.tensor([0, 2]))


def test_hetero_khop_nodes_directional_expansion_avoids_torch_unique(monkeypatch):
    graph = _hetero_graph()

    def fail_unique(*args, **kwargs):
        raise AssertionError("heterogeneous khop directional expansion should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    nodes = khop_nodes(
        graph,
        {"author": torch.tensor([1])},
        num_hops=2,
        direction="out",
        edge_type=("author", "writes", "paper"),
    )

    assert set(nodes) == {"author", "paper"}
    assert torch.equal(nodes["author"], torch.tensor([1]))
    assert torch.equal(nodes["paper"], torch.tensor([0, 2]))


def test_hetero_khop_nodes_directional_expansion_avoids_torch_isin(monkeypatch):
    graph = _hetero_graph()

    def fail_isin(*args, **kwargs):
        raise AssertionError("heterogeneous khop directional expansion should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    nodes = khop_nodes(
        graph,
        {"author": torch.tensor([1])},
        num_hops=2,
        direction="out",
        edge_type=("author", "writes", "paper"),
    )

    assert set(nodes) == {"author", "paper"}
    assert torch.equal(nodes["author"], torch.tensor([1]))
    assert torch.equal(nodes["paper"], torch.tensor([0, 2]))


def test_hetero_khop_subgraph_returns_relation_local_subgraph():
    graph = _hetero_graph()

    subgraph = khop_subgraph(
        graph,
        {"author": torch.tensor([1])},
        num_hops=1,
        direction="out",
        edge_type=("author", "writes", "paper"),
    )

    assert set(subgraph.nodes) == {"author", "paper"}
    assert set(subgraph.edges) == {("author", "writes", "paper")}
    assert torch.equal(subgraph.nodes["author"].x, torch.tensor([[20.0]]))
    assert torch.equal(subgraph.nodes["paper"].x, torch.tensor([[1.0], [3.0]]))
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0, 0], [0, 1]]))
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_weight, torch.tensor([2.0, 3.0]))


def test_hetero_khop_nodes_requires_seed_dict():
    graph = _hetero_graph()

    with pytest.raises(ValueError, match="heterogeneous khop_nodes requires seeds keyed by node type"):
        khop_nodes(
            graph,
            torch.tensor([1]),
            num_hops=1,
            direction="out",
            edge_type=("author", "writes", "paper"),
        )
