import torch

from gnn import Graph
from gnn.core.batch import TemporalEventBatch
from gnn.data.sample import TemporalEventRecord


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
