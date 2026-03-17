import torch

from vgl import Graph


def test_hetero_graph_exposes_typed_node_and_edge_stores():
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 8)},
            "author": {"x": torch.randn(2, 8)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            }
        },
    )

    assert graph.schema.node_types == ("author", "paper")
    assert graph.nodes["paper"].x.shape == (3, 8)
    assert graph.edges[("author", "writes", "paper")].edge_index.shape == (2, 2)


def test_temporal_graph_keeps_time_metadata():
    graph = Graph.temporal(
        nodes={"user": {"x": torch.randn(2, 4)}},
        edges={
            ("user", "interacts", "user"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "timestamp": torch.tensor([1, 2]),
            }
        },
        time_attr="timestamp",
    )

    assert graph.schema.time_attr == "timestamp"
    assert torch.equal(
        graph.edges[("user", "interacts", "user")].timestamp,
        torch.tensor([1, 2]),
    )


def test_temporal_single_relation_graph_exposes_default_edge_index_and_edata():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    timestamp = torch.tensor([1, 2])
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(2, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": edge_index,
                "timestamp": timestamp,
            }
        },
        time_attr="timestamp",
    )

    assert torch.equal(graph.edge_index, edge_index)
    assert torch.equal(graph.edata["timestamp"], timestamp)
