import pytest
import torch

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch
from vgl.data.sample import LinkPredictionRecord
from vgl.train.tasks import LinkPredictionTask


def _batch():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )
    return LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
        ]
    )


def test_link_prediction_task_computes_bce_loss():
    task = LinkPredictionTask(target="label")
    logits = torch.randn(2, requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    assert loss.ndim == 0


def test_link_prediction_task_rejects_unsupported_loss():
    with pytest.raises(ValueError, match="Unsupported loss"):
        LinkPredictionTask(target="label", loss="cross_entropy")


def test_link_prediction_task_rejects_shape_mismatch():
    task = LinkPredictionTask(target="label")
    logits = torch.randn(2, 2, requires_grad=True)

    with pytest.raises(ValueError, match="one logit per candidate edge"):
        task.loss(_batch(), logits, stage="train")
