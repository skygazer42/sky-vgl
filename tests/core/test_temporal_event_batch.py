import torch

from vgl import Graph
from vgl.core.batch import TemporalEventBatch
from vgl.data.sample import TemporalEventRecord


def test_temporal_event_batch_tracks_fields_and_history_views():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )
    records = [
        TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3, label=1),
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5, label=0),
    ]

    batch = TemporalEventBatch.from_records(records)
    history = batch.history_graph(0)
    edge_type = ("node", "interacts", "node")

    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))
    assert torch.equal(batch.timestamp, torch.tensor([3, 5]))
    assert torch.equal(batch.labels, torch.tensor([1, 0]))
    assert torch.equal(history.edges[edge_type].timestamp, torch.tensor([1, 3]))


def test_temporal_event_batch_history_graph_avoids_tensor_item(monkeypatch):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )
    records = [
        TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3, label=1),
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5, label=0),
    ]

    def fail_item(self):
        raise AssertionError("TemporalEventBatch.history_graph should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    batch = TemporalEventBatch.from_records(records)
    history = batch.history_graph(0)
    edge_type = ("node", "interacts", "node")

    assert torch.equal(history.edges[edge_type].timestamp, torch.tensor([1, 3]))


def test_temporal_event_batch_history_graph_avoids_tensor_int(monkeypatch):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )
    records = [
        TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3, label=1),
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5, label=0),
    ]

    def fail_int(self):
        raise AssertionError("TemporalEventBatch.history_graph should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    batch = TemporalEventBatch.from_records(records)
    history = batch.history_graph(0)
    edge_type = ("node", "interacts", "node")

    assert torch.equal(history.edges[edge_type].timestamp, torch.tensor([1, 3]))


def test_temporal_event_batch_avoids_tensor_int_conversion_for_indices(monkeypatch):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )
    records = [
        TemporalEventRecord(
            graph=graph,
            src_index=torch.tensor(0),
            dst_index=torch.tensor(1),
            timestamp=3,
            label=1,
        ),
        TemporalEventRecord(
            graph=graph,
            src_index=torch.tensor(2),
            dst_index=torch.tensor(0),
            timestamp=5,
            label=0,
        ),
    ]

    def fail_int(self):
        raise AssertionError("temporal event batching should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    batch = TemporalEventBatch.from_records(records)

    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))
    assert torch.equal(batch.timestamp, torch.tensor([3, 5]))
    assert torch.equal(batch.labels, torch.tensor([1, 0]))


def test_temporal_event_batch_stacks_event_features():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([1, 4]),
            }
        },
        time_attr="timestamp",
    )
    records = [
        TemporalEventRecord(
            graph=graph,
            src_index=0,
            dst_index=1,
            timestamp=1,
            label=1,
            event_features=torch.tensor([1.0, 0.0]),
        ),
        TemporalEventRecord(
            graph=graph,
            src_index=1,
            dst_index=2,
            timestamp=4,
            label=0,
            event_features=torch.tensor([0.0, 1.0]),
        ),
    ]

    batch = TemporalEventBatch.from_records(records)

    assert torch.equal(batch.event_features, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_temporal_event_batch_batches_multiple_temporal_graphs():
    g1 = Graph.temporal(
        nodes={"node": {"x": torch.randn(2, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0], [1]]),
                "timestamp": torch.tensor([1]),
            }
        },
        time_attr="timestamp",
    )
    g2 = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([2, 4]),
            }
        },
        time_attr="timestamp",
    )

    batch = TemporalEventBatch.from_records(
        [
            TemporalEventRecord(graph=g1, src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=g2, src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )
    edge_type = ("node", "interacts", "node")

    assert torch.equal(batch.src_index, torch.tensor([0, 3]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 4]))
    assert torch.equal(batch.graph.edges[edge_type].edge_index, torch.tensor([[0, 2, 3], [1, 3, 4]]))
    assert torch.equal(batch.graph.edges[edge_type].timestamp, torch.tensor([1, 2, 4]))

HETERO_WRITES = ("author", "writes", "paper")
HETERO_CITES = ("paper", "cites", "paper")


def _hetero_temporal_graph():
    return Graph.temporal(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            HETERO_WRITES: {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([1, 4]),
            },
            HETERO_CITES: {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([2, 5]),
            },
        },
        time_attr="timestamp",
    )


def test_temporal_event_batch_batches_hetero_records_for_single_edge_type():
    g1 = Graph.temporal(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            HETERO_WRITES: {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([1, 4]),
            }
        },
        time_attr="timestamp",
    )
    g2 = Graph.temporal(
        nodes={
            "author": {"x": torch.randn(1, 4)},
            "paper": {"x": torch.randn(2, 4)},
        },
        edges={
            HETERO_WRITES: {
                "edge_index": torch.tensor([[0], [1]]),
                "timestamp": torch.tensor([2]),
            }
        },
        time_attr="timestamp",
    )

    batch = TemporalEventBatch.from_records(
        [
            TemporalEventRecord(graph=g1, src_index=0, dst_index=1, timestamp=1, label=1, edge_type=HETERO_WRITES),
            TemporalEventRecord(graph=g2, src_index=0, dst_index=1, timestamp=2, label=0, edge_type=HETERO_WRITES),
        ]
    )

    assert batch.edge_type == HETERO_WRITES
    assert batch.edge_types == (HETERO_WRITES,)
    assert torch.equal(batch.edge_type_index, torch.tensor([0, 0]))
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 4]))
    assert torch.equal(batch.graph.edges[HETERO_WRITES].edge_index, torch.tensor([[0, 1, 2], [1, 2, 4]]))
    assert torch.equal(batch.history_graph(0).edges[HETERO_WRITES].timestamp, torch.tensor([1]))


def test_temporal_event_batch_supports_mixed_hetero_edge_types():
    graph = _hetero_temporal_graph()

    batch = TemporalEventBatch.from_records(
        [
            TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=4, label=1, edge_type=HETERO_WRITES),
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=5, label=0, edge_type=HETERO_CITES),
        ]
    )

    assert batch.edge_type is None
    assert batch.src_node_type is None
    assert batch.dst_node_type is None
    assert batch.edge_types == (HETERO_WRITES, HETERO_CITES)
    assert torch.equal(batch.edge_type_index, torch.tensor([0, 1]))
    assert torch.equal(batch.src_index, torch.tensor([0, 1]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 2]))
