import warnings

import pytest
import torch
import torch.nn.functional as F

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


def _scalar(tensor):
    return tensor.detach().item()


def test_link_prediction_task_computes_bce_loss():
    task = LinkPredictionTask(target="label")
    logits = torch.randn(2, requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    assert loss.ndim == 0


def test_link_prediction_task_computes_focal_loss():
    task = LinkPredictionTask(target="label", loss="focal", focal_gamma=2.0)
    logits = torch.tensor([2.0, -1.0], requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    targets = torch.tensor([1.0, 0.0])
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    expected = (((1.0 - pt) ** 2.0) * bce).mean()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert _scalar(loss) == pytest.approx(_scalar(expected))


def test_link_prediction_task_applies_pos_weight():
    task = LinkPredictionTask(target="label", pos_weight=3.0)
    logits = torch.tensor([2.0, -1.0], requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    expected = F.binary_cross_entropy_with_logits(
        logits,
        torch.tensor([1.0, 0.0]),
        pos_weight=torch.tensor([3.0]),
    )
    assert _scalar(loss) == pytest.approx(_scalar(expected))


def test_link_prediction_task_applies_pos_weight_to_focal_loss():
    task = LinkPredictionTask(target="label", loss="focal", focal_gamma=2.0, pos_weight=3.0)
    logits = torch.tensor([2.0, -1.0], requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    targets = torch.tensor([1.0, 0.0])
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor([3.0]),
    )
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    expected = (((1.0 - pt) ** 2.0) * bce).mean()
    assert _scalar(loss) == pytest.approx(_scalar(expected))


def test_link_prediction_task_rejects_unsupported_loss():
    with pytest.raises(ValueError, match="Unsupported loss"):
        LinkPredictionTask(target="label", loss="cross_entropy")

    with pytest.raises(ValueError, match="focal_gamma"):
        LinkPredictionTask(target="label", focal_gamma=-1.0)

    with pytest.raises(ValueError, match="pos_weight"):
        LinkPredictionTask(target="label", pos_weight=0.0)


def test_link_prediction_task_rejects_shape_mismatch():
    task = LinkPredictionTask(target="label")
    logits = torch.randn(2, 2, requires_grad=True)

    with pytest.raises(ValueError, match="one logit per candidate edge"):
        task.loss(_batch(), logits, stage="train")
