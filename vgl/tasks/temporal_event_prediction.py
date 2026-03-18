import torch.nn.functional as F

from vgl.tasks.base import Task
from vgl.tasks.losses import balanced_softmax_cross_entropy
from vgl.tasks.losses import ldam_cross_entropy
from vgl.tasks.losses import logit_adjusted_cross_entropy
from vgl.tasks.losses import normalize_class_count
from vgl.tasks.losses import focal_cross_entropy
from vgl.tasks.losses import normalize_class_weight


class TemporalEventPredictionTask(Task):
    def __init__(
        self,
        target="label",
        loss="cross_entropy",
        metrics=None,
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
        self.loss_name = loss
        self.metrics = metrics or []
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

    def loss(self, batch, logits, stage):
        del stage
        class_weight = None
        if self.class_weight is not None:
            class_weight = self.class_weight.to(device=logits.device, dtype=logits.dtype)
        class_count = None
        if self.class_count is not None:
            class_count = self.class_count.to(device=logits.device, dtype=logits.dtype)
        if self.loss_name == "focal":
            return focal_cross_entropy(
                logits,
                batch.labels,
                gamma=self.focal_gamma,
                label_smoothing=self.label_smoothing,
                class_weight=class_weight,
            )
        if self.loss_name == "balanced_softmax":
            if class_count is None:
                raise ValueError("balanced_softmax requires class_count")
            return balanced_softmax_cross_entropy(
                logits,
                batch.labels,
                class_count=class_count,
                label_smoothing=self.label_smoothing,
            )
        if self.loss_name == "ldam":
            return ldam_cross_entropy(
                logits,
                batch.labels,
                class_count=class_count,
                max_margin=self.ldam_max_margin,
                class_weight=class_weight,
                label_smoothing=self.label_smoothing,
            )
        if self.loss_name == "logit_adjustment":
            return logit_adjusted_cross_entropy(
                logits,
                batch.labels,
                class_count=class_count,
                tau=self.logit_adjust_tau,
                class_weight=class_weight,
                label_smoothing=self.label_smoothing,
            )
        return F.cross_entropy(
            logits,
            batch.labels,
            weight=class_weight,
            label_smoothing=self.label_smoothing,
        )

    def targets(self, batch, stage):
        del stage
        return batch.labels
