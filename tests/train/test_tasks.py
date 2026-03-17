import pytest
import torch
import torch.nn.functional as F

from vgl import Graph
from vgl.train.tasks import GraphClassificationTask, NodeClassificationTask, RDropTask


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


def test_node_classification_task_computes_loss():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "train_mask", "train_mask"),
    )
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(graph, logits, stage="train")

    assert loss.ndim == 0


def test_node_classification_task_applies_label_smoothing():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "train_mask", "train_mask"),
        label_smoothing=0.2,
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(graph, logits, stage="train")

    expected = F.cross_entropy(
        logits,
        torch.tensor([0, 1]),
        label_smoothing=0.2,
    )
    assert loss.item() == pytest.approx(expected.item())


def test_node_classification_task_computes_focal_loss():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "train_mask", "train_mask"),
        loss="focal",
        focal_gamma=2.0,
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(graph, logits, stage="train")

    ce = F.cross_entropy(logits, torch.tensor([0, 1]), reduction="none")
    pt = torch.softmax(logits, dim=-1).gather(-1, torch.tensor([[0], [1]])).squeeze(-1)
    expected = (((1 - pt) ** 2.0) * ce).mean()
    assert loss.item() == pytest.approx(expected.item())


def test_node_classification_task_applies_class_weight():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "train_mask", "train_mask"),
        class_weight=[1.0, 3.0],
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(graph, logits, stage="train")

    expected = F.cross_entropy(logits, torch.tensor([0, 1]), weight=torch.tensor([1.0, 3.0]))
    assert loss.item() == pytest.approx(expected.item())


def test_node_classification_task_computes_balanced_softmax_loss():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "train_mask", "train_mask"),
        loss="balanced_softmax",
        class_count=[10.0, 2.0],
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(graph, logits, stage="train")

    balanced_logits = logits + torch.log(torch.tensor([10.0, 2.0]))
    expected = F.cross_entropy(balanced_logits, torch.tensor([0, 1]))
    assert loss.item() == pytest.approx(expected.item())


def test_node_classification_task_computes_ldam_loss():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "train_mask", "train_mask"),
        loss="ldam",
        class_count=[16.0, 1.0],
        ldam_max_margin=0.4,
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(graph, logits, stage="train")

    expected = _expected_ldam_loss(
        logits,
        torch.tensor([0, 1]),
        [16.0, 1.0],
        max_margin=0.4,
    )
    assert loss.item() == pytest.approx(expected.item())


def test_node_classification_task_computes_logit_adjustment_loss():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "train_mask", "train_mask"),
        loss="logit_adjustment",
        class_count=[8.0, 2.0],
        logit_adjust_tau=1.2,
        label_smoothing=0.1,
    )
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]], requires_grad=True)

    loss = task.loss(graph, logits, stage="train")

    expected = _expected_logit_adjustment_loss(
        logits,
        torch.tensor([0, 1]),
        [8.0, 2.0],
        tau=1.2,
        label_smoothing=0.1,
    )
    assert loss.item() == pytest.approx(expected.item())


def test_node_classification_task_rejects_invalid_loss_and_label_smoothing():
    with pytest.raises(ValueError, match="Unsupported loss"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            loss="mse",
        )

    with pytest.raises(ValueError, match="label_smoothing"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            label_smoothing=-0.1,
        )

    with pytest.raises(ValueError, match="label_smoothing"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            label_smoothing=1.0,
        )

    with pytest.raises(ValueError, match="focal_gamma"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            focal_gamma=-1.0,
        )

    with pytest.raises(ValueError, match="class_weight"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            class_weight=[],
        )

    with pytest.raises(ValueError, match="class_count"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            class_count=[],
        )

    with pytest.raises(ValueError, match="ldam requires class_count"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            loss="ldam",
        )

    with pytest.raises(ValueError, match="ldam_max_margin"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            loss="ldam",
            class_count=[4.0, 2.0],
            ldam_max_margin=0.0,
        )

    with pytest.raises(ValueError, match="logit_adjustment requires class_count"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            loss="logit_adjustment",
        )

    with pytest.raises(ValueError, match="logit_adjust_tau"):
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            loss="logit_adjustment",
            class_count=[4.0, 2.0],
            logit_adjust_tau=-0.1,
        )


def test_node_classification_task_uses_configured_split_keys():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(3, 4),
        y=torch.tensor([0, 1, 2]),
        train_nodes=torch.tensor([True, False, False]),
        valid_nodes=torch.tensor([False, True, False]),
        heldout_nodes=torch.tensor([False, False, True]),
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_nodes", "valid_nodes", "heldout_nodes"),
    )
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )

    train_targets = task.targets(graph, stage="train")
    val_targets = task.targets(graph, stage="val")
    test_predictions = task.predictions_for_metrics(graph, logits, stage="test")

    assert train_targets.tolist() == [0]
    assert val_targets.tolist() == [1]
    assert test_predictions.tolist() == [[0.0, 0.0, 10.0]]


class _RDropBatch:
    labels = torch.tensor([0])
    metadata = [{"label": 0}]


def test_rdrop_task_computes_average_supervised_and_consistency_loss():
    base_task = GraphClassificationTask(target="label", label_source="graph")
    task = RDropTask(base_task, alpha=0.5)
    batch = _RDropBatch()
    logits_a = torch.tensor([[2.0, -1.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, 1.0]], requires_grad=True)

    loss = task.paired_loss(batch, logits_a, logits_b, stage="train")

    ce_a = base_task.loss(batch, logits_a, stage="train")
    ce_b = base_task.loss(batch, logits_b, stage="train")
    kl_ab = F.kl_div(
        F.log_softmax(logits_a, dim=-1),
        F.softmax(logits_b, dim=-1),
        reduction="batchmean",
    )
    kl_ba = F.kl_div(
        F.log_softmax(logits_b, dim=-1),
        F.softmax(logits_a, dim=-1),
        reduction="batchmean",
    )
    expected = 0.5 * (ce_a + ce_b) + 0.25 * (kl_ab + kl_ba)
    assert loss.item() == pytest.approx(expected.item())


def test_rdrop_task_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="alpha"):
        RDropTask(GraphClassificationTask(target="label", label_source="graph"), alpha=-0.1)

    with pytest.raises(ValueError, match="cross_entropy"):
        RDropTask(
            base_task=type("UnsupportedTask", (), {"loss_name": "binary_cross_entropy", "metrics": []})(),
            alpha=1.0,
        )
