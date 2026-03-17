import torch.nn.functional as F

from vgl.tasks.base import Task
from vgl.tasks.losses import balanced_softmax_cross_entropy
from vgl.tasks.losses import ldam_cross_entropy
from vgl.tasks.losses import logit_adjusted_cross_entropy
from vgl.tasks.losses import normalize_class_count
from vgl.tasks.losses import focal_cross_entropy
from vgl.tasks.losses import normalize_class_weight


class NodeClassificationTask(Task):
    def __init__(
        self,
        target,
        split,
        loss="cross_entropy",
        metrics=None,
        node_type=None,
        label_smoothing=0.0,
        focal_gamma=2.0,
        class_weight=None,
        class_count=None,
        ldam_max_margin=0.5,
        logit_adjust_tau=1.0,
    ):
        if loss not in {"cross_entropy", "focal", "balanced_softmax", "ldam", "logit_adjustment"}:
            raise ValueError(f"Unsupported loss: {loss}")
        if label_smoothing < 0.0 or label_smoothing >= 1.0:
            raise ValueError("label_smoothing must be in [0.0, 1.0)")
        if focal_gamma < 0.0:
            raise ValueError("focal_gamma must be >= 0")
        self.target = target
        self.train_key, self.val_key, self.test_key = split
        self.loss_name = loss
        self.metrics = metrics or []
        self.node_type = node_type
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.class_weight = normalize_class_weight(class_weight)
        self.class_count = normalize_class_count(class_count)
        self.ldam_max_margin = float(ldam_max_margin)
        self.logit_adjust_tau = float(logit_adjust_tau)
        if self.ldam_max_margin <= 0.0:
            raise ValueError("ldam_max_margin must be > 0")
        if self.logit_adjust_tau < 0.0:
            raise ValueError("logit_adjust_tau must be >= 0")
        if self.loss_name == "ldam" and self.class_count is None:
            raise ValueError("ldam requires class_count")
        if self.loss_name == "logit_adjustment" and self.class_count is None:
            raise ValueError("logit_adjustment requires class_count")

    def _mask_key(self, stage):
        return {
            "train": self.train_key,
            "val": self.val_key,
            "test": self.test_key,
        }.get(stage, f"{stage}_mask")

    def _node_data(self, graph):
        if self.node_type is not None:
            return graph.nodes[self.node_type].data
        if "node" in graph.nodes:
            return graph.nodes["node"].data
        if len(graph.nodes) == 1:
            return next(iter(graph.nodes.values())).data
        raise ValueError("node_type is required for multi-type node classification")

    def loss(self, graph, logits, stage):
        logits = self.predictions_for_metrics(graph, logits, stage)
        targets = self.targets(graph, stage)
        class_weight = None
        if self.class_weight is not None:
            class_weight = self.class_weight.to(device=logits.device, dtype=logits.dtype)
        class_count = None
        if self.class_count is not None:
            class_count = self.class_count.to(device=logits.device, dtype=logits.dtype)
        if self.loss_name == "focal":
            return focal_cross_entropy(
                logits,
                targets,
                gamma=self.focal_gamma,
                label_smoothing=self.label_smoothing,
                class_weight=class_weight,
            )
        if self.loss_name == "balanced_softmax":
            if class_count is None:
                raise ValueError("balanced_softmax requires class_count")
            return balanced_softmax_cross_entropy(
                logits,
                targets,
                class_count=class_count,
                label_smoothing=self.label_smoothing,
            )
        if self.loss_name == "ldam":
            return ldam_cross_entropy(
                logits,
                targets,
                class_count=class_count,
                max_margin=self.ldam_max_margin,
                class_weight=class_weight,
                label_smoothing=self.label_smoothing,
            )
        if self.loss_name == "logit_adjustment":
            return logit_adjusted_cross_entropy(
                logits,
                targets,
                class_count=class_count,
                tau=self.logit_adjust_tau,
                class_weight=class_weight,
                label_smoothing=self.label_smoothing,
            )
        return F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            weight=class_weight,
        )

    def targets(self, graph, stage):
        node_data = self._node_data(graph)
        mask = node_data[self._mask_key(stage)]
        target = node_data[self.target]
        return target[mask]

    def predictions_for_metrics(self, graph, predictions, stage):
        node_data = self._node_data(graph)
        mask = node_data[self._mask_key(stage)]
        return predictions[mask]
