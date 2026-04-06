import pytest
import torch

from vgl import Graph
import vgl.ops as graph_ops
from vgl.ops import line_graph, metapath_reachable_graph


def test_line_graph_turns_edges_into_nodes_and_connects_composable_edges():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    transformed = line_graph(graph, copy_edata=False)

    assert torch.equal(transformed.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(transformed.edge_index, torch.tensor([[0, 0, 2], [1, 2, 0]]))


def test_line_graph_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    def fail_tolist(self):
        raise AssertionError("line_graph should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    transformed = line_graph(graph, copy_edata=False)

    assert torch.equal(transformed.edge_index, torch.tensor([[0, 0, 2], [1, 2, 0]]))


def test_line_graph_avoids_tensor_item(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    def fail_item(self):
        raise AssertionError("line_graph should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    transformed = line_graph(graph, copy_edata=False)

    assert torch.equal(transformed.edge_index, torch.tensor([[0, 0, 2], [1, 2, 0]]))


def test_line_graph_avoids_repeat_interleave(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("line_graph should avoid repeat_interleave interval expansion")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    transformed = line_graph(graph, copy_edata=False)

    assert torch.equal(transformed.edge_index, torch.tensor([[0, 0, 2], [1, 2, 0]]))


def test_line_graph_can_drop_immediate_backtracking_pairs():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    transformed = line_graph(graph, backtracking=False, copy_edata=False)

    assert torch.equal(transformed.edge_index, torch.tensor([[0], [2]]))


def test_line_graph_copy_edata_promotes_edge_tensors_to_node_features():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_data={
            "weight": torch.tensor([1.5, 2.5]),
            "timestamp": torch.tensor([4, 7]),
        },
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    transformed = line_graph(graph, copy_edata=True)

    assert torch.equal(transformed.n_id, torch.tensor([0, 1]))
    assert torch.equal(transformed.weight, torch.tensor([1.5, 2.5]))
    assert torch.equal(transformed.timestamp, torch.tensor([4, 7]))


def test_metapath_reachable_graph_deduplicates_hetero_reachable_pairs():
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 0, 1], [0, 1, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    transformed = metapath_reachable_graph(graph, [writes, published_in])
    edge_type = ("author", "writes__published_in", "venue")

    assert set(transformed.nodes) == {"author", "venue"}
    assert set(transformed.edges) == {edge_type}
    assert torch.equal(transformed.nodes["author"].x, graph.nodes["author"].x)
    assert torch.equal(transformed.nodes["venue"].x, graph.nodes["venue"].x)
    assert torch.equal(transformed.edges[edge_type].edge_index, torch.tensor([[0, 1], [0, 0]]))


def test_metapath_reachable_graph_avoids_tensor_tolist(monkeypatch):
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 0, 1], [0, 1, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    def fail_tolist(self):
        raise AssertionError("metapath_reachable_graph should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    transformed = metapath_reachable_graph(graph, [writes, published_in])
    edge_type = ("author", "writes__published_in", "venue")

    assert torch.equal(transformed.edges[edge_type].edge_index, torch.tensor([[0, 1], [0, 0]]))


def test_metapath_reachable_graph_avoids_tensor_item(monkeypatch):
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 0, 1], [0, 1, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    def fail_item(self):
        raise AssertionError("metapath_reachable_graph should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    transformed = metapath_reachable_graph(graph, [writes, published_in])
    edge_type = ("author", "writes__published_in", "venue")

    assert torch.equal(transformed.edges[edge_type].edge_index, torch.tensor([[0, 1], [0, 0]]))


def test_metapath_reachable_graph_avoids_repeat_interleave(monkeypatch):
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 0, 1], [0, 1, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("metapath_reachable_graph should avoid repeat_interleave interval expansion")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    transformed = metapath_reachable_graph(graph, [writes, published_in])
    edge_type = ("author", "writes__published_in", "venue")

    assert torch.equal(transformed.edges[edge_type].edge_index, torch.tensor([[0, 1], [0, 0]]))


def test_metapath_reachable_graph_avoids_torch_unique(monkeypatch):
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 0, 1], [0, 1, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    def fail_unique(*args, **kwargs):
        raise AssertionError("metapath_reachable_graph should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    transformed = metapath_reachable_graph(graph, [writes, published_in])
    edge_type = ("author", "writes__published_in", "venue")

    assert torch.equal(transformed.edges[edge_type].edge_index, torch.tensor([[0, 1], [0, 0]]))


def test_metapath_reachable_graph_supports_single_node_type_multi_relation():
    follows = ("node", "follows", "node")
    likes = ("node", "likes", "node")
    graph = Graph.hetero(
        nodes={"node": {"x": torch.tensor([[0.0], [1.0], [2.0], [3.0]])}},
        edges={
            follows: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            likes: {"edge_index": torch.tensor([[1, 2], [3, 0]])},
        },
    )

    transformed = metapath_reachable_graph(graph, [follows, likes])
    edge_type = ("node", "follows__likes", "node")

    assert set(transformed.nodes) == {"node"}
    assert set(transformed.edges) == {edge_type}
    assert torch.equal(transformed.nodes["node"].x, graph.nodes["node"].x)
    assert torch.equal(transformed.edges[edge_type].edge_index, torch.tensor([[0, 1], [3, 0]]))


def test_metapath_reachable_graph_rejects_non_composable_edge_types():
    writes = ("author", "writes", "paper")
    hosted_by = ("venue", "hosted_by", "conference")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0]])},
            "paper": {"x": torch.tensor([[2.0]])},
            "venue": {"x": torch.tensor([[3.0]])},
            "conference": {"x": torch.tensor([[4.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0], [0]])},
            hosted_by: {"edge_index": torch.tensor([[0], [0]])},
        },
    )

    with pytest.raises(ValueError, match="compose"):
        metapath_reachable_graph(graph, [writes, hosted_by])


def test_random_walk_follows_homogeneous_edges_and_includes_seeds():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    assert hasattr(graph_ops, "random_walk")
    traces = graph_ops.random_walk(graph, torch.tensor([0, 2]), length=3)

    assert torch.equal(traces, torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]]))


def test_random_walk_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    def fail_tolist(self):
        raise AssertionError("random_walk should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    traces = graph_ops.random_walk(graph, torch.tensor([0, 2]), length=3)

    assert torch.equal(traces, torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]]))


def test_random_walk_avoids_tensor_item(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    def fail_item(self):
        raise AssertionError("random_walk should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    traces = graph_ops.random_walk(graph, torch.tensor([0, 2]), length=3)

    assert torch.equal(traces, torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]]))


def test_random_walk_avoids_torch_unique(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    def fail_unique(*args, **kwargs):
        raise AssertionError("random_walk should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    traces = graph_ops.random_walk(graph, torch.tensor([0, 2]), length=3)

    assert torch.equal(traces, torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]]))


def test_random_walk_pads_with_negative_one_after_dead_end():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([[1.0], [2.0]]),
    )

    assert hasattr(graph_ops, "random_walk")
    traces = graph_ops.random_walk(graph, torch.tensor([0, 1]), length=3)

    assert torch.equal(traces, torch.tensor([[0, 1, -1, -1], [1, -1, -1, -1]]))


def test_random_walk_supports_single_node_type_multi_relation_with_selected_edge_type():
    follows = ("node", "follows", "node")
    likes = ("node", "likes", "node")
    graph = Graph.hetero(
        nodes={"node": {"x": torch.tensor([[0.0], [1.0], [2.0], [3.0]])}},
        edges={
            follows: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            likes: {"edge_index": torch.tensor([[0, 2], [3, 1]])},
        },
    )

    assert hasattr(graph_ops, "random_walk")
    traces = graph_ops.random_walk(graph, torch.tensor([0, 1]), length=2, edge_type=follows)

    assert torch.equal(traces, torch.tensor([[0, 1, 2], [1, 2, -1]]))


def test_random_walk_rejects_multi_step_non_composable_relation():
    writes = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[3.0], [4.0]])},
        },
        edges={writes: {"edge_index": torch.tensor([[0, 1], [0, 1]])}},
    )

    assert hasattr(graph_ops, "random_walk")
    with pytest.raises(ValueError, match="compose"):
        graph_ops.random_walk(graph, torch.tensor([0]), length=2, edge_type=writes)


def test_metapath_random_walk_follows_typed_relation_sequence():
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [0, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    assert hasattr(graph_ops, "metapath_random_walk")
    traces = graph_ops.metapath_random_walk(graph, torch.tensor([0, 1]), [writes, published_in])

    assert torch.equal(traces, torch.tensor([[0, 0, 0], [1, 1, 0]]))


def test_metapath_random_walk_avoids_tensor_tolist(monkeypatch):
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [0, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    def fail_tolist(self):
        raise AssertionError("metapath_random_walk should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    traces = graph_ops.metapath_random_walk(graph, torch.tensor([0, 1]), [writes, published_in])

    assert torch.equal(traces, torch.tensor([[0, 0, 0], [1, 1, 0]]))


def test_metapath_random_walk_avoids_tensor_item(monkeypatch):
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [0, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    def fail_item(self):
        raise AssertionError("metapath_random_walk should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    traces = graph_ops.metapath_random_walk(graph, torch.tensor([0, 1]), [writes, published_in])

    assert torch.equal(traces, torch.tensor([[0, 0, 0], [1, 1, 0]]))


def test_metapath_random_walk_pads_after_missing_relation_step():
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [0, 1]])},
            published_in: {"edge_index": torch.tensor([[0], [0]])},
        },
    )

    assert hasattr(graph_ops, "metapath_random_walk")
    traces = graph_ops.metapath_random_walk(graph, torch.tensor([0, 1]), [writes, published_in])

    assert torch.equal(traces, torch.tensor([[0, 0, 0], [1, 1, -1]]))


def test_metapath_random_walk_rejects_non_composable_edge_types():
    writes = ("author", "writes", "paper")
    hosted_by = ("venue", "hosted_by", "conference")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0]])},
            "paper": {"x": torch.tensor([[2.0]])},
            "venue": {"x": torch.tensor([[3.0]])},
            "conference": {"x": torch.tensor([[4.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0], [0]])},
            hosted_by: {"edge_index": torch.tensor([[0], [0]])},
        },
    )

    assert hasattr(graph_ops, "metapath_random_walk")
    with pytest.raises(ValueError, match="compose"):
        graph_ops.metapath_random_walk(graph, torch.tensor([0]), [writes, hosted_by])
