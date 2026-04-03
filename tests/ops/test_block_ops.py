import pytest
import torch

from tests.pinning import assert_tensor_pin_state
from vgl import Graph
from vgl.graph import Block, GraphSchema, HeteroBlock
from vgl.ops import to_block, to_hetero_block
from vgl.storage import FeatureStore, InMemoryGraphStore


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


def test_to_block_avoids_torch_isin_scans(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 1], [1, 2, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("to_block should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    block = to_block(graph, torch.tensor([1, 2]))

    assert torch.equal(block.edata["e_id"], torch.tensor([0, 1, 2]))
    assert torch.equal(block.edge_index, torch.tensor([[2, 0, 1], [0, 1, 0]]))


def test_to_block_uses_graph_store_counts_for_featureless_storage_backed_graph():
    edge_type = ("node", "to", "node")
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(edge_type,),
        node_features={"node": ()},
        edge_features={edge_type: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {edge_type: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    block = to_block(graph, torch.tensor([3]))

    assert torch.equal(block.dst_n_id, torch.tensor([3]))
    assert torch.equal(block.src_n_id, torch.tensor([3]))
    assert torch.equal(block.srcdata["n_id"], torch.tensor([3]))
    assert torch.equal(block.dstdata["n_id"], torch.tensor([3]))
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
    assert_tensor_pin_state(pinned.src_n_id)
    assert_tensor_pin_state(pinned.dst_n_id)
    assert_tensor_pin_state(pinned.srcdata["x"])
    assert_tensor_pin_state(pinned.edge_index)
    assert not block.src_n_id.is_pinned()


def test_to_hetero_block_builds_multi_relation_block_layer():
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
                "edge_index": torch.tensor([[0, 1, 2], [2, 2, 0]]),
                "weight": torch.tensor([5.0, 6.0, 7.0]),
            },
        },
    )

    block = to_hetero_block(graph, {"paper": torch.tensor([0, 2])})

    assert isinstance(block, HeteroBlock)
    assert block.edge_types == (writes, cites)
    assert block.src_store_types == {"author": "author", "paper": "paper__src"}
    assert block.dst_store_types == {"paper": "paper__dst"}
    assert set(block.graph.nodes) == {"author", "paper__src", "paper__dst"}
    assert set(block.graph.edges) == {
        ("author", "writes", "paper__dst"),
        ("paper__src", "cites", "paper__dst"),
    }
    assert torch.equal(block.src_n_id["author"], torch.tensor([1]))
    assert torch.equal(block.src_n_id["paper"], torch.tensor([0, 2, 1]))
    assert torch.equal(block.dst_n_id["paper"], torch.tensor([0, 2]))
    assert torch.equal(block.srcdata("author")["x"], torch.tensor([[20.0]]))
    assert torch.equal(block.srcdata("paper")["x"], torch.tensor([[1.0], [3.0], [2.0]]))
    assert torch.equal(block.dstdata("paper")["x"], torch.tensor([[1.0], [3.0]]))
    assert torch.equal(block.edata(writes)["e_id"], torch.tensor([1, 2]))
    assert torch.equal(block.edata(writes)["weight"], torch.tensor([2.0, 3.0]))
    assert torch.equal(block.edge_index(writes), torch.tensor([[0, 0], [0, 1]]))
    assert torch.equal(block.edata(cites)["e_id"], torch.tensor([0, 1, 2]))
    assert torch.equal(block.edata(cites)["weight"], torch.tensor([5.0, 6.0, 7.0]))
    assert torch.equal(block.edge_index(cites), torch.tensor([[0, 2, 1], [1, 1, 0]]))


def test_to_hetero_block_avoids_torch_isin_scans(monkeypatch):
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
                "edge_index": torch.tensor([[0, 1, 2], [2, 2, 0]]),
                "weight": torch.tensor([5.0, 6.0, 7.0]),
            },
        },
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("to_hetero_block should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    block = to_hetero_block(graph, {"paper": torch.tensor([0, 2])})

    assert torch.equal(block.edata(writes)["e_id"], torch.tensor([1, 2]))
    assert torch.equal(block.edata(cites)["e_id"], torch.tensor([0, 1, 2]))


def test_to_hetero_block_respects_relation_subset():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]])},
            cites: {"edge_index": torch.tensor([[0, 1, 2], [2, 2, 0]])},
        },
    )

    block = to_hetero_block(graph, {"paper": torch.tensor([0, 2])}, edge_types=(writes,))

    assert block.edge_types == (writes,)
    assert block.src_store_types == {"author": "author"}
    assert block.dst_store_types == {"paper": "paper"}
    assert set(block.graph.nodes) == {"author", "paper"}
    assert set(block.graph.edges) == {("author", "writes", "paper")}
    assert torch.equal(block.src_n_id["author"], torch.tensor([1]))
    assert torch.equal(block.dst_n_id["paper"], torch.tensor([0, 2]))


def test_to_hetero_block_keeps_schema_stable_when_one_destination_frontier_is_empty():
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]])},
            written_by: {"edge_index": torch.tensor([[0, 2], [0, 1]])},
        },
    )

    block = to_hetero_block(
        graph,
        {
            "paper": torch.tensor([0]),
            "author": torch.empty((0,), dtype=torch.long),
        },
    )

    assert block.edge_types == (writes, written_by)
    assert block.src_store_types == {"author": "author__src", "paper": "paper__src"}
    assert block.dst_store_types == {"paper": "paper__dst", "author": "author__dst"}
    assert set(block.graph.nodes) == {"author__src", "paper__src", "paper__dst", "author__dst"}
    assert set(block.graph.edges) == {
        ("author__src", "writes", "paper__dst"),
        ("paper__src", "written_by", "author__dst"),
    }
    assert torch.equal(block.src_n_id["author"], torch.tensor([1]))
    assert torch.equal(block.src_n_id["paper"], torch.empty((0,), dtype=torch.long))
    assert torch.equal(block.dst_n_id["paper"], torch.tensor([0]))
    assert torch.equal(block.dst_n_id["author"], torch.empty((0,), dtype=torch.long))
    assert torch.equal(block.edata(writes)["e_id"], torch.tensor([1]))
    assert torch.equal(block.edge_index(writes), torch.tensor([[0], [0]], dtype=torch.long))
    assert torch.equal(block.edata(written_by)["e_id"], torch.empty((0,), dtype=torch.long))
    assert torch.equal(block.edge_index(written_by), torch.empty((2, 0), dtype=torch.long))


def test_hetero_block_to_moves_graph_and_metadata():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]])},
            cites: {"edge_index": torch.tensor([[0, 1, 2], [2, 2, 0]])},
        },
    )

    block = to_hetero_block(graph, {"paper": torch.tensor([0, 2])})
    moved = block.to(device=_transfer_device(), dtype=torch.float64)

    assert moved is not block
    assert moved.src_n_id["author"].device.type == "meta"
    assert moved.src_n_id["paper"].device.type == "meta"
    assert moved.dst_n_id["paper"].device.type == "meta"
    assert moved.srcdata("paper")["x"].device.type == "meta"
    assert moved.srcdata("paper")["x"].dtype == torch.float64
    assert moved.edge_index(cites).device.type == "meta"


def test_hetero_block_pin_memory_pins_graph_and_metadata():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]])},
            cites: {"edge_index": torch.tensor([[0, 1, 2], [2, 2, 0]])},
        },
    )

    block = to_hetero_block(graph, {"paper": torch.tensor([0, 2])})
    pinned = block.pin_memory()

    assert pinned is not block
    assert_tensor_pin_state(pinned.src_n_id["author"])
    assert_tensor_pin_state(pinned.src_n_id["paper"])
    assert_tensor_pin_state(pinned.dst_n_id["paper"])
    assert_tensor_pin_state(pinned.srcdata("paper")["x"])
    assert_tensor_pin_state(pinned.edge_index(cites))
    assert not block.src_n_id["author"].is_pinned()
