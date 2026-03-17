import torch

from vgl import Graph


def test_snapshot_filters_temporal_edges_without_copying_features():
    graph = Graph.temporal(
        nodes={"user": {"x": torch.randn(2, 4)}},
        edges={
            ("user", "interacts", "user"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "edge_attr": torch.tensor([[1.0], [3.0]]),
                "timestamp": torch.tensor([1, 3]),
            }
        },
        time_attr="timestamp",
    )

    snapshot = graph.snapshot(2)

    assert snapshot.schema.time_attr == "timestamp"
    assert snapshot.edges[("user", "interacts", "user")].edge_index.shape[1] == 1
    assert torch.equal(snapshot.edges[("user", "interacts", "user")].edge_attr, torch.tensor([[1.0]]))
    assert snapshot.nodes["user"].x.data_ptr() == graph.nodes["user"].x.data_ptr()


def test_window_filters_edges_inside_time_range():
    graph = Graph.temporal(
        nodes={"user": {"x": torch.randn(2, 4)}},
        edges={
            ("user", "interacts", "user"): {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 1]]),
                "edge_weight": torch.tensor([1.0, 2.0, 3.0]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )

    window = graph.window(start=2, end=4)

    assert window.edges[("user", "interacts", "user")].edge_index.shape[1] == 1
    assert torch.equal(
        window.edges[("user", "interacts", "user")].timestamp,
        torch.tensor([3]),
    )
    assert torch.equal(
        window.edges[("user", "interacts", "user")].edge_weight,
        torch.tensor([2.0]),
    )
