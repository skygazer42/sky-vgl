import pytest
import torch
import torch.nn.functional as F

from vgl import Graph
from vgl.train.tasks import (
    BootstrapTask,
    ConfidencePenaltyTask,
    FloodingTask,
    GeneralizedCrossEntropyTask,
    GraphClassificationTask,
    LinkPredictionTask,
    NodeClassificationTask,
    Poly1CrossEntropyTask,
    RDropTask,
    SymmetricCrossEntropyTask,
)


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


def _expected_symmetric_cross_entropy(
    logits,
    targets,
    *,
    alpha=1.0,
    beta=1.0,
    label_clip=1e-4,
):
    if logits.ndim == 1:
        positive = torch.sigmoid(logits)
        probs = torch.stack([1.0 - positive, positive], dim=-1)
        target_probs = torch.stack(
            [
                1.0 - targets.to(dtype=logits.dtype),
                targets.to(dtype=logits.dtype),
            ],
            dim=-1,
        )
        ce = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype))
    else:
        probs = torch.softmax(logits, dim=-1)
        target_probs = torch.full_like(probs, label_clip)
        target_probs[torch.arange(targets.numel()), targets] = 1.0
        ce = F.cross_entropy(logits, targets)
    target_probs = target_probs.clamp_min(label_clip)
    rce = -(probs * torch.log(target_probs)).sum(dim=-1).mean()
    return alpha * ce + beta * rce


def _expected_poly1_cross_entropy(logits, targets, *, epsilon=1.0, base_loss=None):
    if base_loss is None:
        if logits.ndim == 1:
            base_loss = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype))
        else:
            base_loss = F.cross_entropy(logits, targets)
    if logits.ndim == 1:
        probs = torch.sigmoid(logits)
        true_probs = torch.where(targets.to(dtype=logits.dtype) > 0.5, probs, 1.0 - probs)
    else:
        probs = torch.softmax(logits, dim=-1)
        true_probs = probs.gather(-1, targets.to(dtype=torch.long).view(-1, 1)).squeeze(-1)
    return base_loss + epsilon * (1.0 - true_probs).mean()


def _expected_bootstrap_cross_entropy(logits, targets, *, beta=0.95, mode="soft"):
    if logits.ndim == 1:
        probs = torch.sigmoid(logits)
        if mode == "soft":
            pseudo_targets = probs
        else:
            pseudo_targets = (probs >= 0.5).to(dtype=logits.dtype)
        blended_targets = beta * targets.to(dtype=logits.dtype) + (1.0 - beta) * pseudo_targets
        return F.binary_cross_entropy_with_logits(logits, blended_targets)
    probs = torch.softmax(logits, dim=-1)
    observed_targets = torch.zeros_like(probs)
    observed_targets[torch.arange(targets.numel()), targets.to(dtype=torch.long)] = 1.0
    if mode == "soft":
        pseudo_targets = probs
    else:
        pseudo_targets = torch.zeros_like(probs)
        pseudo_targets.scatter_(1, probs.argmax(dim=-1, keepdim=True), 1.0)
    blended_targets = beta * observed_targets + (1.0 - beta) * pseudo_targets
    return -(blended_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


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


class _BinaryBatch:
    labels = torch.tensor([1.0, 0.0])


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


def test_flooding_task_applies_flooded_loss_only_during_training():
    base_task = GraphClassificationTask(target="label", label_source="graph")
    task = FloodingTask(base_task, level=0.3)
    batch = _RDropBatch()
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)

    train_loss = task.loss(batch, logits, stage="train")
    val_loss = task.loss(batch, logits, stage="val")
    base_loss = base_task.loss(batch, logits, stage="train")

    expected = (base_loss - 0.3).abs() + 0.3
    assert train_loss.item() == pytest.approx(expected.item())
    assert val_loss.item() == pytest.approx(base_loss.item())


def test_flooding_task_wraps_paired_loss_when_base_task_supports_it():
    base_task = RDropTask(GraphClassificationTask(target="label", label_source="graph"), alpha=0.5)
    task = FloodingTask(base_task, level=0.4)
    batch = _RDropBatch()
    logits_a = torch.tensor([[2.0, -1.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, 1.0]], requires_grad=True)

    loss = task.paired_loss(batch, logits_a, logits_b, stage="train")
    base_loss = base_task.paired_loss(batch, logits_a, logits_b, stage="train")
    expected = (base_loss - 0.4).abs() + 0.4

    assert loss.item() == pytest.approx(expected.item())


def test_flooding_task_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="level"):
        FloodingTask(GraphClassificationTask(target="label", label_source="graph"), level=-0.1)


def test_confidence_penalty_task_adds_entropy_regularization_only_during_training():
    base_task = GraphClassificationTask(target="label", label_source="graph")
    task = ConfidencePenaltyTask(base_task, coefficient=0.2)
    batch = _RDropBatch()
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)

    train_loss = task.loss(batch, logits, stage="train")
    val_loss = task.loss(batch, logits, stage="val")
    base_loss = base_task.loss(batch, logits, stage="train")
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
    expected = base_loss - 0.2 * entropy

    assert train_loss.item() == pytest.approx(expected.item())
    assert val_loss.item() == pytest.approx(base_loss.item())


def test_confidence_penalty_task_supports_binary_logits():
    base_task = LinkPredictionTask(target="label")
    task = ConfidencePenaltyTask(base_task, coefficient=0.1)
    batch = _BinaryBatch()
    logits = torch.tensor([1.5, -0.5], requires_grad=True)

    loss = task.loss(batch, logits, stage="train")

    base_loss = base_task.loss(batch, logits, stage="train")
    probs = torch.sigmoid(logits)
    entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs)).mean()
    expected = base_loss - 0.1 * entropy

    assert loss.item() == pytest.approx(expected.item())


def test_confidence_penalty_task_wraps_paired_loss_when_base_task_supports_it():
    base_task = RDropTask(GraphClassificationTask(target="label", label_source="graph"), alpha=0.5)
    task = ConfidencePenaltyTask(base_task, coefficient=0.15)
    batch = _RDropBatch()
    logits_a = torch.tensor([[2.0, -1.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, 1.0]], requires_grad=True)

    loss = task.paired_loss(batch, logits_a, logits_b, stage="train")

    base_loss = base_task.paired_loss(batch, logits_a, logits_b, stage="train")
    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    entropy_a = -(probs_a * torch.log(probs_a)).sum(dim=-1).mean()
    entropy_b = -(probs_b * torch.log(probs_b)).sum(dim=-1).mean()
    expected = base_loss - 0.15 * 0.5 * (entropy_a + entropy_b)

    assert loss.item() == pytest.approx(expected.item())


def test_confidence_penalty_task_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="coefficient"):
        ConfidencePenaltyTask(GraphClassificationTask(target="label", label_source="graph"), coefficient=-0.1)


def test_generalized_cross_entropy_task_applies_gce_only_during_training():
    base_task = GraphClassificationTask(target="label", label_source="graph")
    task = GeneralizedCrossEntropyTask(base_task, q=0.7)
    batch = _RDropBatch()
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)

    train_loss = task.loss(batch, logits, stage="train")
    val_loss = task.loss(batch, logits, stage="val")
    base_loss = base_task.loss(batch, logits, stage="train")
    pt = torch.softmax(logits, dim=-1)[0, 0]
    expected = (1.0 - pt.pow(0.7)) / 0.7

    assert train_loss.item() == pytest.approx(expected.item())
    assert val_loss.item() == pytest.approx(base_loss.item())


def test_generalized_cross_entropy_task_supports_binary_logits():
    base_task = LinkPredictionTask(target="label")
    task = GeneralizedCrossEntropyTask(base_task, q=0.5)
    batch = _BinaryBatch()
    logits = torch.tensor([1.5, -0.5], requires_grad=True)

    loss = task.loss(batch, logits, stage="train")

    probs = torch.sigmoid(logits)
    true_probs = torch.stack([probs[0], 1.0 - probs[1]])
    expected = ((1.0 - true_probs.pow(0.5)) / 0.5).mean()

    assert loss.item() == pytest.approx(expected.item())


def test_generalized_cross_entropy_task_wraps_paired_loss_and_preserves_extra_regularization():
    base_task = RDropTask(GraphClassificationTask(target="label", label_source="graph"), alpha=0.5)
    task = GeneralizedCrossEntropyTask(base_task, q=0.7)
    batch = _RDropBatch()
    logits_a = torch.tensor([[2.0, -1.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, 1.0]], requires_grad=True)

    loss = task.paired_loss(batch, logits_a, logits_b, stage="train")

    base_paired = base_task.paired_loss(batch, logits_a, logits_b, stage="train")
    base_supervised = 0.5 * (
        base_task.loss(batch, logits_a, stage="train") + base_task.loss(batch, logits_b, stage="train")
    )
    probs_a = torch.softmax(logits_a, dim=-1)[0, 0]
    probs_b = torch.softmax(logits_b, dim=-1)[0, 0]
    gce_a = (1.0 - probs_a.pow(0.7)) / 0.7
    gce_b = (1.0 - probs_b.pow(0.7)) / 0.7
    expected = (base_paired - base_supervised) + 0.5 * (gce_a + gce_b)

    assert loss.item() == pytest.approx(expected.item())


def test_generalized_cross_entropy_task_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="q"):
        GeneralizedCrossEntropyTask(GraphClassificationTask(target="label", label_source="graph"), q=0.0)

    with pytest.raises(ValueError, match="q"):
        GeneralizedCrossEntropyTask(GraphClassificationTask(target="label", label_source="graph"), q=1.1)


def test_symmetric_cross_entropy_task_applies_sce_only_during_training():
    base_task = GraphClassificationTask(target="label", label_source="graph")
    task = SymmetricCrossEntropyTask(base_task, alpha=0.7, beta=0.3, label_clip=1e-4)
    batch = _RDropBatch()
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)

    train_loss = task.loss(batch, logits, stage="train")
    val_loss = task.loss(batch, logits, stage="val")
    base_loss = base_task.loss(batch, logits, stage="train")
    expected = _expected_symmetric_cross_entropy(
        logits,
        torch.tensor([0]),
        alpha=0.7,
        beta=0.3,
        label_clip=1e-4,
    )

    assert train_loss.item() == pytest.approx(expected.item())
    assert val_loss.item() == pytest.approx(base_loss.item())


def test_symmetric_cross_entropy_task_supports_binary_logits():
    base_task = LinkPredictionTask(target="label")
    task = SymmetricCrossEntropyTask(base_task, alpha=0.6, beta=0.4, label_clip=1e-4)
    batch = _BinaryBatch()
    logits = torch.tensor([1.5, -0.5], requires_grad=True)

    loss = task.loss(batch, logits, stage="train")

    expected = _expected_symmetric_cross_entropy(
        logits,
        torch.tensor([1.0, 0.0]),
        alpha=0.6,
        beta=0.4,
        label_clip=1e-4,
    )

    assert loss.item() == pytest.approx(expected.item())


def test_symmetric_cross_entropy_task_wraps_paired_loss_and_preserves_extra_regularization():
    base_task = RDropTask(GraphClassificationTask(target="label", label_source="graph"), alpha=0.5)
    task = SymmetricCrossEntropyTask(base_task, alpha=0.7, beta=0.3, label_clip=1e-4)
    batch = _RDropBatch()
    logits_a = torch.tensor([[2.0, -1.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, 1.0]], requires_grad=True)

    loss = task.paired_loss(batch, logits_a, logits_b, stage="train")

    base_paired = base_task.paired_loss(batch, logits_a, logits_b, stage="train")
    base_supervised = 0.5 * (
        base_task.loss(batch, logits_a, stage="train") + base_task.loss(batch, logits_b, stage="train")
    )
    sce_a = _expected_symmetric_cross_entropy(
        logits_a,
        torch.tensor([0]),
        alpha=0.7,
        beta=0.3,
        label_clip=1e-4,
    )
    sce_b = _expected_symmetric_cross_entropy(
        logits_b,
        torch.tensor([0]),
        alpha=0.7,
        beta=0.3,
        label_clip=1e-4,
    )
    expected = (base_paired - base_supervised) + 0.5 * (sce_a + sce_b)

    assert loss.item() == pytest.approx(expected.item())


def test_symmetric_cross_entropy_task_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="alpha"):
        SymmetricCrossEntropyTask(GraphClassificationTask(target="label", label_source="graph"), alpha=-0.1)

    with pytest.raises(ValueError, match="beta"):
        SymmetricCrossEntropyTask(GraphClassificationTask(target="label", label_source="graph"), beta=-0.1)

    with pytest.raises(ValueError, match="label_clip"):
        SymmetricCrossEntropyTask(GraphClassificationTask(target="label", label_source="graph"), label_clip=0.0)

    with pytest.raises(ValueError, match="label_clip"):
        SymmetricCrossEntropyTask(GraphClassificationTask(target="label", label_source="graph"), label_clip=1.1)


def test_poly1_cross_entropy_task_applies_poly1_only_during_training():
    base_task = GraphClassificationTask(target="label", label_source="graph")
    task = Poly1CrossEntropyTask(base_task, epsilon=0.7)
    batch = _RDropBatch()
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)

    train_loss = task.loss(batch, logits, stage="train")
    val_loss = task.loss(batch, logits, stage="val")
    base_loss = base_task.loss(batch, logits, stage="train")
    expected = _expected_poly1_cross_entropy(
        logits,
        torch.tensor([0]),
        epsilon=0.7,
        base_loss=base_loss,
    )

    assert train_loss.item() == pytest.approx(expected.item())
    assert val_loss.item() == pytest.approx(base_loss.item())


def test_poly1_cross_entropy_task_supports_binary_logits():
    base_task = LinkPredictionTask(target="label")
    task = Poly1CrossEntropyTask(base_task, epsilon=0.5)
    batch = _BinaryBatch()
    logits = torch.tensor([1.5, -0.5], requires_grad=True)

    loss = task.loss(batch, logits, stage="train")

    expected = _expected_poly1_cross_entropy(
        logits,
        torch.tensor([1.0, 0.0]),
        epsilon=0.5,
        base_loss=base_task.loss(batch, logits, stage="train"),
    )

    assert loss.item() == pytest.approx(expected.item())


def test_poly1_cross_entropy_task_wraps_paired_loss_and_preserves_extra_regularization():
    base_task = RDropTask(GraphClassificationTask(target="label", label_source="graph"), alpha=0.5)
    task = Poly1CrossEntropyTask(base_task, epsilon=0.7)
    batch = _RDropBatch()
    logits_a = torch.tensor([[2.0, -1.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, 1.0]], requires_grad=True)

    loss = task.paired_loss(batch, logits_a, logits_b, stage="train")

    base_paired = base_task.paired_loss(batch, logits_a, logits_b, stage="train")
    base_supervised = 0.5 * (
        base_task.loss(batch, logits_a, stage="train") + base_task.loss(batch, logits_b, stage="train")
    )
    poly1_a = _expected_poly1_cross_entropy(
        logits_a,
        torch.tensor([0]),
        epsilon=0.7,
        base_loss=base_task.loss(batch, logits_a, stage="train"),
    )
    poly1_b = _expected_poly1_cross_entropy(
        logits_b,
        torch.tensor([0]),
        epsilon=0.7,
        base_loss=base_task.loss(batch, logits_b, stage="train"),
    )
    expected = (base_paired - base_supervised) + 0.5 * (poly1_a + poly1_b)

    assert loss.item() == pytest.approx(expected.item())


def test_poly1_cross_entropy_task_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="epsilon"):
        Poly1CrossEntropyTask(GraphClassificationTask(target="label", label_source="graph"), epsilon=-0.1)


def test_bootstrap_task_applies_soft_bootstrap_only_during_training():
    base_task = GraphClassificationTask(target="label", label_source="graph")
    task = BootstrapTask(base_task, beta=0.8, mode="soft")
    batch = _RDropBatch()
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)

    train_loss = task.loss(batch, logits, stage="train")
    val_loss = task.loss(batch, logits, stage="val")
    base_loss = base_task.loss(batch, logits, stage="train")
    expected = _expected_bootstrap_cross_entropy(
        logits,
        torch.tensor([0]),
        beta=0.8,
        mode="soft",
    )

    assert train_loss.item() == pytest.approx(expected.item())
    assert val_loss.item() == pytest.approx(base_loss.item())


def test_bootstrap_task_supports_binary_logits_with_hard_mode():
    base_task = LinkPredictionTask(target="label")
    task = BootstrapTask(base_task, beta=0.6, mode="hard")
    batch = _BinaryBatch()
    logits = torch.tensor([1.5, -0.5], requires_grad=True)

    loss = task.loss(batch, logits, stage="train")

    expected = _expected_bootstrap_cross_entropy(
        logits,
        torch.tensor([1.0, 0.0]),
        beta=0.6,
        mode="hard",
    )

    assert loss.item() == pytest.approx(expected.item())


def test_bootstrap_task_wraps_paired_loss_and_preserves_extra_regularization():
    base_task = RDropTask(GraphClassificationTask(target="label", label_source="graph"), alpha=0.5)
    task = BootstrapTask(base_task, beta=0.8, mode="soft")
    batch = _RDropBatch()
    logits_a = torch.tensor([[2.0, -1.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, 1.0]], requires_grad=True)

    loss = task.paired_loss(batch, logits_a, logits_b, stage="train")

    base_paired = base_task.paired_loss(batch, logits_a, logits_b, stage="train")
    base_supervised = 0.5 * (
        base_task.loss(batch, logits_a, stage="train") + base_task.loss(batch, logits_b, stage="train")
    )
    bootstrap_a = _expected_bootstrap_cross_entropy(
        logits_a,
        torch.tensor([0]),
        beta=0.8,
        mode="soft",
    )
    bootstrap_b = _expected_bootstrap_cross_entropy(
        logits_b,
        torch.tensor([0]),
        beta=0.8,
        mode="soft",
    )
    expected = (base_paired - base_supervised) + 0.5 * (bootstrap_a + bootstrap_b)

    assert loss.item() == pytest.approx(expected.item())


def test_bootstrap_task_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="beta"):
        BootstrapTask(GraphClassificationTask(target="label", label_source="graph"), beta=-0.1)

    with pytest.raises(ValueError, match="beta"):
        BootstrapTask(GraphClassificationTask(target="label", label_source="graph"), beta=1.1)

    with pytest.raises(ValueError, match="mode"):
        BootstrapTask(GraphClassificationTask(target="label", label_source="graph"), mode="median")
