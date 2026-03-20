import torch

from vgl import Graph
from vgl.graph.stores import EdgeStore, NodeStore


def test_node_store_to_moves_tensors_and_preserves_non_tensors():
    x = torch.randn(3, 4)
    labels = ["a", "b", "c"]
    store = NodeStore(type_name="node", data={"x": x, "labels": labels})

    moved = store.to(device="cpu")

    assert moved is not store
    assert moved.type_name == store.type_name
    assert moved.data is not store.data
    assert moved.x.device.type == "cpu"
    assert torch.equal(moved.x, x)
    assert moved.data["labels"] is labels
    assert store.x.data_ptr() == x.data_ptr()


def test_edge_store_pin_memory_pins_tensors_and_preserves_non_tensors():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    meta = {"source": "synthetic"}
    store = EdgeStore(
        type_name=("node", "to", "node"),
        data={"edge_index": edge_index, "meta": meta},
    )

    pinned = store.pin_memory()

    assert pinned is not store
    assert pinned.type_name == store.type_name
    assert pinned.data is not store.data
    assert pinned.edge_index.is_pinned()
    assert torch.equal(pinned.edge_index, edge_index)
    assert pinned.data["meta"] is meta
    assert not store.edge_index.is_pinned()


def test_graph_to_moves_tensors_for_homo_hetero_and_temporal_graphs():
    homo = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        edge_data={"edge_weight": torch.tensor([1.0, 2.0]), "name": "homo"},
    )
    hetero = Graph.hetero(
        nodes={"user": {"x": torch.randn(2, 3), "kind": "user"}},
        edges={
            ("user", "follows", "user"): {
                "edge_index": torch.tensor([[0], [1]]),
                "edge_attr": torch.tensor([[1.0]]),
                "label": "hetero",
            }
        },
    )
    temporal = Graph.temporal(
        nodes={"node": {"x": torch.randn(2, 3), "tag": "temporal"}},
        edges={
            ("node", "to", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "timestamp": torch.tensor([1.0, 2.0]),
            }
        },
        time_attr="timestamp",
    )

    homo_edge_dtype = homo.edata["edge_weight"].dtype
    hetero_edge_dtype = hetero.edges[("user", "follows", "user")].edge_attr.dtype
    temporal_edge_dtype = temporal.edges[("node", "to", "node")].timestamp.dtype

    moved_homo = homo.to(device="cpu", dtype=torch.float64)
    moved_hetero = hetero.to(device="cpu", dtype=torch.float64)
    moved_temporal = temporal.to(device="cpu", dtype=torch.float64)

    assert moved_homo is not homo
    assert moved_homo.schema == homo.schema
    assert moved_homo.x.device.type == "cpu"
    assert moved_homo.edges[("node", "to", "node")] is not homo.edges[("node", "to", "node")]
    assert moved_homo.edata["edge_weight"].dtype == torch.float64
    assert moved_homo.edata["edge_weight"].data_ptr() != homo.edata["edge_weight"].data_ptr()
    assert homo.edata["edge_weight"].dtype == homo_edge_dtype
    assert moved_homo.edata["name"] is homo.edata["name"]

    assert moved_hetero is not hetero
    assert moved_hetero.schema == hetero.schema
    assert moved_hetero.nodes["user"].x.device.type == "cpu"
    assert moved_hetero.edges[("user", "follows", "user")] is not hetero.edges[("user", "follows", "user")]
    assert moved_hetero.edges[("user", "follows", "user")].edge_attr.dtype == torch.float64
    assert (
        moved_hetero.edges[("user", "follows", "user")].edge_attr.data_ptr()
        != hetero.edges[("user", "follows", "user")].edge_attr.data_ptr()
    )
    assert hetero.edges[("user", "follows", "user")].edge_attr.dtype == hetero_edge_dtype
    assert (
        moved_hetero.edges[("user", "follows", "user")].data["label"]
        is hetero.edges[("user", "follows", "user")].data["label"]
    )

    assert moved_temporal is not temporal
    assert moved_temporal.schema == temporal.schema
    assert moved_temporal.schema.time_attr == "timestamp"
    assert moved_temporal.nodes["node"].x.device.type == "cpu"
    assert moved_temporal.edges[("node", "to", "node")] is not temporal.edges[("node", "to", "node")]
    assert moved_temporal.edges[("node", "to", "node")].timestamp.dtype == torch.float64
    assert (
        moved_temporal.edges[("node", "to", "node")].timestamp.data_ptr()
        != temporal.edges[("node", "to", "node")].timestamp.data_ptr()
    )
    assert temporal.edges[("node", "to", "node")].timestamp.dtype == temporal_edge_dtype
    assert moved_temporal.nodes["node"].data["tag"] is temporal.nodes["node"].data["tag"]


def test_graph_view_to_moves_visible_tensors_without_mutating_base():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4), "group": "all"}},
        edges={
            ("node", "to", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1.0, 2.0, 3.0]),
                "edge_weight": torch.tensor([0.1, 0.2, 0.3]),
            }
        },
        time_attr="timestamp",
    )
    view = graph.snapshot(2)
    base_dtype = graph.nodes["node"].x.dtype
    base_timestamp_dtype = graph.edges[("node", "to", "node")].timestamp.dtype
    base_edge_weight_dtype = graph.edges[("node", "to", "node")].edge_weight.dtype

    moved = view.to(device="cpu", dtype=torch.float64)

    assert moved is not view
    assert moved.base is graph
    assert moved.schema == view.schema
    assert moved.nodes["node"].x.device.type == "cpu"
    assert moved.nodes["node"].x.dtype == torch.float64
    assert moved.nodes["node"].data["group"] is view.nodes["node"].data["group"]
    assert moved.edges[("node", "to", "node")] is not view.edges[("node", "to", "node")]
    assert moved.edges[("node", "to", "node")].edge_index.shape[1] == 2
    assert moved.edges[("node", "to", "node")].timestamp.dtype == torch.float64
    assert moved.edges[("node", "to", "node")].edge_weight.dtype == torch.float64
    assert (
        moved.edges[("node", "to", "node")].edge_weight.data_ptr()
        != view.edges[("node", "to", "node")].edge_weight.data_ptr()
    )
    assert graph.nodes["node"].x.dtype == base_dtype
    assert graph.edges[("node", "to", "node")].timestamp.dtype == base_timestamp_dtype
    assert graph.edges[("node", "to", "node")].edge_weight.dtype == base_edge_weight_dtype
