import pytest
import torch
import torch.nn.functional as F

from vgl.train.tasks import GraphClassificationTask


class FakeBatch:
    labels = torch.tensor([1, 0])
    metadata = [{"label": 1}, {"label": 0}]


def _expected_ldam_loss(logits, targets, class_count, *, max_margin=0.5, label_smoothing=0.0):
    margins = torch.pow(torch.tensor(class_count, dtype=logits.dtype), -0.25)
    margins = margins * (max_margin / margins.max())
    adjusted_logits = logits.clone()
    adjusted_logits[torch.arange(targets.numel()), targets] -= margins[targets]
    return F.cross_entropy(
        adjusted_logits,
        targets,
        label_smoothing=label_smoothing,
    )


def _expected_logit_adjustment_loss(
    logits,
    targets,
    class_count,
    *,
    tau=1.0,
    label_smoothing=0.0,
    class_weight=None,
):
    class_prior = torch.tensor(class_count, dtype=logits.dtype)
    class_prior = class_prior / class_prior.sum()
    adjusted_logits = logits + tau * torch.log(class_prior)
    weight = None
    if class_weight is not None:
        weight = torch.tensor(class_weight, dtype=logits.dtype)
    return F.cross_entropy(
        adjusted_logits,
        targets,
        label_smoothing=label_smoothing,
        weight=weight,
    )


def test_graph_classification_task_uses_batch_labels():
    task = GraphClassificationTask(target="y", label_source="graph")
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    assert loss.ndim == 0


def test_graph_classification_task_uses_metadata_labels():
    task = GraphClassificationTask(target="label", label_source="metadata")
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    assert loss.ndim == 0


def test_graph_classification_task_applies_label_smoothing():
    task = GraphClassificationTask(target="label", label_source="metadata", label_smoothing=0.2)
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    expected = F.cross_entropy(
        logits,
        torch.tensor([1, 0]),
        label_smoothing=0.2,
    )
    assert loss.item() == pytest.approx(expected.item())


def test_graph_classification_task_computes_focal_loss():
    task = GraphClassificationTask(target="label", label_source="metadata", loss="focal", focal_gamma=2.0)
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    ce = F.cross_entropy(logits, torch.tensor([1, 0]), reduction="none")
    pt = torch.softmax(logits, dim=-1).gather(-1, torch.tensor([[1], [0]])).squeeze(-1)
    expected = (((1 - pt) ** 2.0) * ce).mean()
    assert loss.item() == pytest.approx(expected.item())


def test_graph_classification_task_applies_class_weight_to_focal_loss():
    task = GraphClassificationTask(
        target="label",
        label_source="metadata",
        loss="focal",
        focal_gamma=2.0,
        class_weight=[1.0, 3.0],
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    ce = F.cross_entropy(
        logits,
        torch.tensor([1, 0]),
        reduction="none",
        weight=torch.tensor([1.0, 3.0]),
    )
    pt = torch.softmax(logits, dim=-1).gather(-1, torch.tensor([[1], [0]])).squeeze(-1)
    expected = (((1 - pt) ** 2.0) * ce).mean()
    assert loss.item() == pytest.approx(expected.item())


def test_graph_classification_task_computes_balanced_softmax_loss():
    task = GraphClassificationTask(
        target="label",
        label_source="metadata",
        loss="balanced_softmax",
        class_count=[10.0, 2.0],
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    balanced_logits = logits + torch.log(torch.tensor([10.0, 2.0]))
    expected = F.cross_entropy(balanced_logits, torch.tensor([1, 0]))
    assert loss.item() == pytest.approx(expected.item())


def test_graph_classification_task_computes_ldam_loss():
    task = GraphClassificationTask(
        target="label",
        label_source="metadata",
        loss="ldam",
        class_count=[16.0, 1.0],
        ldam_max_margin=0.4,
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    expected = _expected_ldam_loss(
        logits,
        torch.tensor([1, 0]),
        [16.0, 1.0],
        max_margin=0.4,
    )
    assert loss.item() == pytest.approx(expected.item())


def test_graph_classification_task_computes_logit_adjustment_loss():
    task = GraphClassificationTask(
        target="label",
        label_source="metadata",
        loss="logit_adjustment",
        class_count=[8.0, 2.0],
        logit_adjust_tau=1.2,
        class_weight=[1.0, 3.0],
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    expected = _expected_logit_adjustment_loss(
        logits,
        torch.tensor([1, 0]),
        [8.0, 2.0],
        tau=1.2,
        class_weight=[1.0, 3.0],
    )
    assert loss.item() == pytest.approx(expected.item())


def test_graph_classification_task_rejects_invalid_loss_and_label_smoothing():
    with pytest.raises(ValueError, match="Unsupported loss"):
        GraphClassificationTask(target="label", label_source="graph", loss="mse")

    with pytest.raises(ValueError, match="label_smoothing"):
        GraphClassificationTask(target="label", label_source="graph", label_smoothing=-0.1)

    with pytest.raises(ValueError, match="label_smoothing"):
        GraphClassificationTask(target="label", label_source="graph", label_smoothing=1.0)

    with pytest.raises(ValueError, match="focal_gamma"):
        GraphClassificationTask(target="label", label_source="graph", focal_gamma=-1.0)

    with pytest.raises(ValueError, match="class_weight"):
        GraphClassificationTask(target="label", label_source="graph", class_weight=[])

    with pytest.raises(ValueError, match="class_count"):
        GraphClassificationTask(target="label", label_source="graph", class_count=[])

    with pytest.raises(ValueError, match="ldam requires class_count"):
        GraphClassificationTask(target="label", label_source="graph", loss="ldam")

    with pytest.raises(ValueError, match="ldam_max_margin"):
        GraphClassificationTask(
            target="label",
            label_source="graph",
            loss="ldam",
            class_count=[4.0, 2.0],
            ldam_max_margin=0.0,
        )

    with pytest.raises(ValueError, match="logit_adjustment requires class_count"):
        GraphClassificationTask(target="label", label_source="graph", loss="logit_adjustment")

    with pytest.raises(ValueError, match="logit_adjust_tau"):
        GraphClassificationTask(
            target="label",
            label_source="graph",
            loss="logit_adjustment",
            class_count=[4.0, 2.0],
            logit_adjust_tau=-0.1,
        )
