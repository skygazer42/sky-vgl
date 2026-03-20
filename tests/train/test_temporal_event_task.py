import warnings

import pytest
import torch
import torch.nn.functional as F

from vgl import Graph
from vgl.core.batch import TemporalEventBatch
from vgl.data.sample import TemporalEventRecord
from vgl.train.tasks import TemporalEventPredictionTask


def _scalar(tensor):
    return tensor.detach().item()


def _expected_ldam_loss(logits, targets, class_count, *, max_margin=0.5):
    margins = torch.pow(torch.tensor(class_count, dtype=logits.dtype), -0.25)
    margins = margins * (max_margin / margins.max())
    adjusted_logits = logits.clone()
    adjusted_logits[torch.arange(targets.numel()), targets] -= margins[targets]
    return F.cross_entropy(adjusted_logits, targets)


def _expected_logit_adjustment_loss(logits, targets, class_count, *, tau=1.0, class_weight=None):
    class_prior = torch.tensor(class_count, dtype=logits.dtype)
    class_prior = class_prior / class_prior.sum()
    adjusted_logits = logits + tau * torch.log(class_prior)
    weight = None
    if class_weight is not None:
        weight = torch.tensor(class_weight, dtype=logits.dtype)
    return F.cross_entropy(adjusted_logits, targets, weight=weight)


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


def test_temporal_event_prediction_task_applies_label_smoothing():
    task = TemporalEventPredictionTask(target="label", label_smoothing=0.2)
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    expected = F.cross_entropy(
        logits,
        torch.tensor([1, 0]),
        label_smoothing=0.2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert _scalar(loss) == pytest.approx(_scalar(expected))


def test_temporal_event_prediction_task_computes_focal_loss():
    task = TemporalEventPredictionTask(target="label", loss="focal", focal_gamma=2.0)
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    ce = F.cross_entropy(logits, torch.tensor([1, 0]), reduction="none")
    pt = torch.softmax(logits, dim=-1).gather(-1, torch.tensor([[1], [0]])).squeeze(-1)
    expected = (((1 - pt) ** 2.0) * ce).mean()
    assert _scalar(loss) == pytest.approx(_scalar(expected))


def test_temporal_event_prediction_task_applies_class_weight():
    task = TemporalEventPredictionTask(target="label", class_weight=[1.0, 4.0])
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    expected = F.cross_entropy(logits, torch.tensor([1, 0]), weight=torch.tensor([1.0, 4.0]))
    assert _scalar(loss) == pytest.approx(_scalar(expected))


def test_temporal_event_prediction_task_computes_balanced_softmax_loss():
    task = TemporalEventPredictionTask(target="label", loss="balanced_softmax", class_count=[10.0, 2.0])
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    balanced_logits = logits + torch.log(torch.tensor([10.0, 2.0]))
    expected = F.cross_entropy(balanced_logits, torch.tensor([1, 0]))
    assert _scalar(loss) == pytest.approx(_scalar(expected))


def test_temporal_event_prediction_task_computes_ldam_loss():
    task = TemporalEventPredictionTask(
        target="label",
        loss="ldam",
        class_count=[16.0, 1.0],
        ldam_max_margin=0.4,
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    expected = _expected_ldam_loss(
        logits,
        torch.tensor([1, 0]),
        [16.0, 1.0],
        max_margin=0.4,
    )
    assert _scalar(loss) == pytest.approx(_scalar(expected))


def test_temporal_event_prediction_task_computes_logit_adjustment_loss():
    task = TemporalEventPredictionTask(
        target="label",
        loss="logit_adjustment",
        class_count=[8.0, 2.0],
        logit_adjust_tau=1.2,
        class_weight=[1.0, 3.0],
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    expected = _expected_logit_adjustment_loss(
        logits,
        torch.tensor([1, 0]),
        [8.0, 2.0],
        tau=1.2,
        class_weight=[1.0, 3.0],
    )
    assert _scalar(loss) == pytest.approx(_scalar(expected))


def test_temporal_event_prediction_task_rejects_unsupported_loss():
    with pytest.raises(ValueError, match="Unsupported loss"):
        TemporalEventPredictionTask(target="label", loss="bce")

    with pytest.raises(ValueError, match="label_smoothing"):
        TemporalEventPredictionTask(target="label", label_smoothing=-0.1)

    with pytest.raises(ValueError, match="label_smoothing"):
        TemporalEventPredictionTask(target="label", label_smoothing=1.0)

    with pytest.raises(ValueError, match="focal_gamma"):
        TemporalEventPredictionTask(target="label", focal_gamma=-1.0)

    with pytest.raises(ValueError, match="class_weight"):
        TemporalEventPredictionTask(target="label", class_weight=[])

    with pytest.raises(ValueError, match="class_count"):
        TemporalEventPredictionTask(target="label", class_count=[])

    with pytest.raises(ValueError, match="ldam requires class_count"):
        TemporalEventPredictionTask(target="label", loss="ldam")

    with pytest.raises(ValueError, match="ldam_max_margin"):
        TemporalEventPredictionTask(
            target="label",
            loss="ldam",
            class_count=[4.0, 2.0],
            ldam_max_margin=0.0,
        )

    with pytest.raises(ValueError, match="logit_adjustment requires class_count"):
        TemporalEventPredictionTask(target="label", loss="logit_adjustment")

    with pytest.raises(ValueError, match="logit_adjust_tau"):
        TemporalEventPredictionTask(
            target="label",
            loss="logit_adjustment",
            class_count=[4.0, 2.0],
            logit_adjust_tau=-0.1,
        )

