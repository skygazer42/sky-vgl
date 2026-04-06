import pytest
import torch

from vgl import Graph
from vgl.ops import edge_subgraph, in_subgraph, node_subgraph, out_subgraph
from vgl.ops.subgraph import _lookup_positions, _membership_mask


def test_node_subgraph_filters_edges_and_relabels_nodes():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    subgraph = node_subgraph(graph, torch.tensor([0, 2, 3]))

    assert torch.equal(subgraph.x, torch.tensor([[1.0], [3.0], [4.0]]))
    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))


def test_node_subgraph_homo_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    def fail_tolist(self):
        raise AssertionError("homo node_subgraph should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    subgraph = node_subgraph(graph, torch.tensor([0, 2, 3]))

    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))


def test_node_subgraph_homo_avoids_torch_isin(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("homo node_subgraph should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    subgraph = node_subgraph(graph, torch.tensor([0, 2, 3]))

    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))


def test_node_subgraph_homo_avoids_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    def fail_int(self):
        raise AssertionError("homo node_subgraph should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    subgraph = node_subgraph(graph, torch.tensor([0, 2, 3]))

    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))


def test_node_subgraph_homo_sorted_node_ids_avoid_torch_unique_in_membership(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )
    original_unique = torch.unique

    def guarded_unique(*args, **kwargs):
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_name == "_membership_mask":
            raise AssertionError("sorted unique node_subgraph ids should not be uniqued again in membership checks")
        return original_unique(*args, **kwargs)

    import inspect

    monkeypatch.setattr(torch, "unique", guarded_unique)

    subgraph = node_subgraph(graph, torch.tensor([0, 2, 3]))

    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))


def test_membership_mask_contiguous_ranges_avoid_searchsorted(monkeypatch):
    original_searchsorted = torch.searchsorted

    def guarded_searchsorted(*args, **kwargs):
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_name == "_membership_mask":
            raise AssertionError("contiguous subgraph membership checks should avoid searchsorted")
        return original_searchsorted(*args, **kwargs)

    import inspect

    monkeypatch.setattr(torch, "searchsorted", guarded_searchsorted)

    mask = _membership_mask(
        torch.tensor([0, 2, 3, 5]),
        torch.tensor([1, 2, 3, 4]),
    )

    assert torch.equal(mask, torch.tensor([False, True, True, False]))


def test_lookup_positions_avoids_tensor_item_in_missing_id_errors(monkeypatch):
    def fail_item(self):
        raise AssertionError("subgraph lookup should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    with pytest.raises(KeyError, match="missing node id 7"):
        _lookup_positions(torch.tensor([0, 2, 6]), torch.tensor([7]), entity_name="node")

    with pytest.raises(KeyError, match="missing node id 4"):
        _lookup_positions(torch.tensor([0, 2, 6]), torch.tensor([2, 4]), entity_name="node")


def test_lookup_positions_avoids_tensor_int_in_missing_id_errors(monkeypatch):
    def fail_int(self):
        raise AssertionError("subgraph lookup should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    with pytest.raises(KeyError, match="missing node id 7"):
        _lookup_positions(torch.tensor([0, 2, 6]), torch.tensor([7]), entity_name="node")

    with pytest.raises(KeyError, match="missing node id 4"):
        _lookup_positions(torch.tensor([0, 2, 6]), torch.tensor([2, 4]), entity_name="node")


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


def test_hetero_node_subgraph_filters_relation_and_relabels_per_type():
    graph = Graph.hetero(
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

    subgraph = node_subgraph(
        graph,
        {"author": torch.tensor([1]), "paper": torch.tensor([0, 2])},
        edge_type=("author", "writes", "paper"),
    )

    assert set(subgraph.nodes) == {"author", "paper"}
    assert set(subgraph.edges) == {("author", "writes", "paper")}
    assert torch.equal(subgraph.nodes["author"].x, torch.tensor([[20.0]]))
    assert torch.equal(subgraph.nodes["paper"].x, torch.tensor([[1.0], [3.0]]))
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0, 0], [0, 1]]))
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_weight, torch.tensor([2.0, 3.0]))


def test_node_subgraph_hetero_avoids_tensor_tolist(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "edge_weight": torch.tensor([1.0, 2.0, 3.0]),
            }
        },
    )

    def fail_tolist(self):
        raise AssertionError("hetero node_subgraph should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    subgraph = node_subgraph(
        graph,
        {"author": torch.tensor([1]), "paper": torch.tensor([0, 2])},
        edge_type=("author", "writes", "paper"),
    )

    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0, 0], [0, 1]]))


def test_node_subgraph_hetero_avoids_torch_isin(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "edge_weight": torch.tensor([1.0, 2.0, 3.0]),
            }
        },
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("hetero node_subgraph should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    subgraph = node_subgraph(
        graph,
        {"author": torch.tensor([1]), "paper": torch.tensor([0, 2])},
        edge_type=("author", "writes", "paper"),
    )

    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0, 0], [0, 1]]))


def test_hetero_edge_subgraph_preserves_participating_node_spaces():
    graph = Graph.hetero(
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

    subgraph = edge_subgraph(graph, torch.tensor([0, 2]), edge_type=("author", "writes", "paper"))

    assert set(subgraph.nodes) == {"author", "paper"}
    assert set(subgraph.edges) == {("author", "writes", "paper")}
    assert torch.equal(subgraph.nodes["author"].x, graph.nodes["author"].x)
    assert torch.equal(subgraph.nodes["paper"].x, graph.nodes["paper"].x)
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_weight, torch.tensor([1.0, 3.0]))


def test_in_subgraph_filters_homo_edges_and_preserves_node_space():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"edge_weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    subgraph = in_subgraph(graph, torch.tensor([3, 0]))

    assert torch.equal(subgraph.x, graph.x)
    assert torch.equal(subgraph.edge_index, torch.tensor([[2, 3, 1], [3, 0, 3]]))
    assert torch.equal(subgraph.edata["edge_weight"], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(subgraph.edata["e_id"], torch.tensor([1, 2, 3]))


def test_out_subgraph_filters_homo_edges_and_preserves_node_space():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"edge_weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    subgraph = out_subgraph(graph, torch.tensor([0, 3]))

    assert torch.equal(subgraph.x, graph.x)
    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 3], [2, 0]]))
    assert torch.equal(subgraph.edata["edge_weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(subgraph.edata["e_id"], torch.tensor([0, 2]))


def test_frontier_subgraph_avoids_torch_isin(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("frontier subgraph queries should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    inbound = in_subgraph(graph, torch.tensor([3, 0]))
    outbound = out_subgraph(graph, torch.tensor([0, 3]))

    assert torch.equal(inbound.edge_index, torch.tensor([[2, 3, 1], [3, 0, 3]]))
    assert torch.equal(outbound.edge_index, torch.tensor([[0, 3], [2, 0]]))


def test_frontier_subgraph_avoids_repeat_interleave(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("frontier subgraph queries should avoid repeat_interleave interval expansion")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    inbound = in_subgraph(graph, torch.tensor([3, 0]))
    outbound = out_subgraph(graph, torch.tensor([0, 3]))

    assert torch.equal(inbound.edge_index, torch.tensor([[2, 3, 1], [3, 0, 3]]))
    assert torch.equal(outbound.edge_index, torch.tensor([[0, 3], [2, 0]]))


def test_hetero_in_subgraph_filters_all_relations_by_destination_frontier():
    follows = ("user", "follows", "user")
    plays = ("user", "plays", "game")
    graph = Graph.hetero(
        nodes={
            "user": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "game": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            follows: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "weight": torch.tensor([0.1, 0.2, 0.3]),
            },
            plays: {
                "edge_index": torch.tensor([[0, 1, 1], [0, 0, 2]]),
                "hours": torch.tensor([4.0, 5.0, 6.0]),
            },
        },
    )

    subgraph = in_subgraph(graph, {"user": torch.tensor([0, 2]), "game": torch.tensor([2])})

    assert set(subgraph.nodes) == {"user", "game"}
    assert set(subgraph.edges) == {follows, plays}
    assert torch.equal(subgraph.nodes["user"].x, graph.nodes["user"].x)
    assert torch.equal(subgraph.nodes["game"].x, graph.nodes["game"].x)
    assert torch.equal(subgraph.edges[follows].edge_index, torch.tensor([[1, 2], [2, 0]]))
    assert torch.equal(subgraph.edges[follows].weight, torch.tensor([0.2, 0.3]))
    assert torch.equal(subgraph.edges[follows].e_id, torch.tensor([1, 2]))
    assert torch.equal(subgraph.edges[plays].edge_index, torch.tensor([[1], [2]]))
    assert torch.equal(subgraph.edges[plays].hours, torch.tensor([6.0]))
    assert torch.equal(subgraph.edges[plays].e_id, torch.tensor([2]))


def test_hetero_out_subgraph_filters_all_relations_by_source_frontier():
    follows = ("user", "follows", "user")
    plays = ("user", "plays", "game")
    graph = Graph.hetero(
        nodes={
            "user": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "game": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            follows: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "weight": torch.tensor([0.1, 0.2, 0.3]),
            },
            plays: {
                "edge_index": torch.tensor([[0, 1, 1], [0, 0, 2]]),
                "hours": torch.tensor([4.0, 5.0, 6.0]),
            },
        },
    )

    subgraph = out_subgraph(graph, {"user": torch.tensor([1])})

    assert set(subgraph.nodes) == {"user", "game"}
    assert set(subgraph.edges) == {follows, plays}
    assert torch.equal(subgraph.edges[follows].edge_index, torch.tensor([[1], [2]]))
    assert torch.equal(subgraph.edges[follows].weight, torch.tensor([0.2]))
    assert torch.equal(subgraph.edges[follows].e_id, torch.tensor([1]))
    assert torch.equal(subgraph.edges[plays].edge_index, torch.tensor([[1, 1], [0, 2]]))
    assert torch.equal(subgraph.edges[plays].hours, torch.tensor([5.0, 6.0]))
    assert torch.equal(subgraph.edges[plays].e_id, torch.tensor([1, 2]))


def test_frontier_subgraph_requires_typed_frontiers_for_multi_type_graphs():
    graph = Graph.hetero(
        nodes={
            "user": {"x": torch.tensor([[1.0]])},
            "game": {"x": torch.tensor([[2.0]])},
        },
        edges={
            ("user", "plays", "game"): {"edge_index": torch.tensor([[0], [0]])},
        },
    )

    with pytest.raises(ValueError, match="keyed by node type"):
        in_subgraph(graph, torch.tensor([0]))
