import pytest
import torch

from gnn import Graph
from gnn.core.batch import TemporalEventBatch
from gnn.data.sample import TemporalEventRecord
from gnn.train.tasks import TemporalEventPredictionTask


def _batch():
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
    return TemporalEventBatch.from_records(
        [
            TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )


def test_temporal_event_prediction_task_computes_loss():
    task = TemporalEventPredictionTask(target="label")
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    assert loss.ndim == 0


def test_temporal_event_prediction_task_rejects_unsupported_loss():
    with pytest.raises(ValueError, match="Unsupported loss"):
        TemporalEventPredictionTask(target="label", loss="bce")
