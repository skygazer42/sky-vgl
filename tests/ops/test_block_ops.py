import pytest
import torch

from vgl import Graph
from vgl.graph import Block
from vgl.ops import to_block


def _transfer_device() -> str:
    return "meta"


def test_to_block_builds_homo_block_with_id_metadata_and_feature_slices():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 1], [1, 2, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"weight": torch.tensor([0.5, 0.6, 0.7, 0.8])},
    )

    block = to_block(graph, torch.tensor([1, 2]))

    assert isinstance(block, Block)
    assert block.edge_type == ("node", "to", "node")
    assert block.src_type == "node"
    assert block.dst_type == "node"
    assert torch.equal(block.dst_n_id, torch.tensor([1, 2]))
    assert torch.equal(block.src_n_id, torch.tensor([1, 2, 0]))
    assert torch.equal(block.srcdata["n_id"], torch.tensor([1, 2, 0]))
    assert torch.equal(block.dstdata["n_id"], torch.tensor([1, 2]))
    assert torch.equal(block.srcdata["x"], torch.tensor([[2.0], [3.0], [1.0]]))
    assert torch.equal(block.dstdata["x"], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(block.edata["e_id"], torch.tensor([0, 1, 2]))
    assert torch.equal(block.edata["weight"], torch.tensor([0.5, 0.6, 0.7]))
    assert torch.equal(block.edge_index, torch.tensor([[2, 0, 1], [0, 1, 0]]))


def test_to_block_include_dst_in_src_false_keeps_only_true_predecessors():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    default_block = to_block(graph, torch.tensor([1, 2]))
    block = to_block(graph, torch.tensor([1, 2]), include_dst_in_src=False)

    assert torch.equal(default_block.src_n_id, torch.tensor([1, 2, 0]))
    assert torch.equal(block.dst_n_id, torch.tensor([1, 2]))
    assert torch.equal(block.src_n_id, torch.tensor([0, 1]))
    assert torch.equal(block.edge_index, torch.tensor([[0, 1], [0, 1]]))


def test_to_block_builds_relation_local_hetero_block():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0], [1]]),
            },
        },
    )

    block = to_block(graph, torch.tensor([0, 2]), edge_type=writes)

    assert block.edge_type == writes
    assert block.src_type == "author"
    assert block.dst_type == "paper"
    assert set(block.graph.nodes) == {"author", "paper"}
    assert set(block.graph.edges) == {("author", "writes", "paper")}
    assert torch.equal(block.src_n_id, torch.tensor([1]))
    assert torch.equal(block.dst_n_id, torch.tensor([0, 2]))
    assert torch.equal(block.srcdata["x"], torch.tensor([[20.0]]))
    assert torch.equal(block.dstdata["x"], torch.tensor([[1.0], [3.0]]))
    assert torch.equal(block.edata["e_id"], torch.tensor([1, 2]))
    assert torch.equal(block.edata["weight"], torch.tensor([2.0, 3.0]))
    assert torch.equal(block.edge_index, torch.tensor([[0, 0], [0, 1]]))


def test_to_block_handles_empty_incoming_frontier():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    block = to_block(graph, torch.tensor([0]))

    assert torch.equal(block.dst_n_id, torch.tensor([0]))
    assert torch.equal(block.src_n_id, torch.tensor([0]))
    assert torch.equal(block.edata["e_id"], torch.empty((0,), dtype=torch.long))
    assert torch.equal(block.edge_index, torch.empty((2, 0), dtype=torch.long))


def test_to_block_rejects_ambiguous_relation_selection():
    graph = Graph.hetero(
        nodes={"node": {"x": torch.tensor([[1.0], [2.0], [3.0]])}},
        edges={
            ("node", "follows", "node"): {"edge_index": torch.tensor([[0], [1]])},
            ("node", "likes", "node"): {"edge_index": torch.tensor([[1], [2]])},
        },
    )

    with pytest.raises(ValueError, match="edge_type"):
        to_block(graph, torch.tensor([1]))


def test_to_block_rejects_out_of_range_destination_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([[1.0], [2.0]]),
    )

    with pytest.raises(ValueError, match="destination"):
        to_block(graph, torch.tensor([2]))


def test_block_to_moves_graph_and_metadata():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    block = to_block(graph, torch.tensor([1, 2]))
    moved = block.to(device=_transfer_device(), dtype=torch.float64)

    assert moved is not block
    assert moved.graph.nodes[next(iter(moved.graph.nodes))].data["n_id"].device.type == "meta"
    assert moved.src_n_id.device.type == "meta"
    assert moved.dst_n_id.device.type == "meta"
    assert moved.srcdata["x"].device.type == "meta"
    assert moved.srcdata["x"].dtype == torch.float64
    assert moved.edge_index.device.type == "meta"


def test_block_pin_memory_pins_graph_and_metadata():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    block = to_block(graph, torch.tensor([1, 2]))
    pinned = block.pin_memory()

    assert pinned is not block
    assert pinned.src_n_id.is_pinned()
    assert pinned.dst_n_id.is_pinned()
    assert pinned.srcdata["x"].is_pinned()
    assert pinned.edge_index.is_pinned()
    assert not block.src_n_id.is_pinned()
