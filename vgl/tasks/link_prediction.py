import torch.nn.functional as F

from vgl.tasks.base import Task
from vgl.tasks.losses import focal_binary_cross_entropy_with_logits
from vgl.tasks.losses import normalize_pos_weight


class LinkPredictionTask(Task):
    def __init__(
        self,
        target="label",
        loss="binary_cross_entropy",
        metrics=None,
        focal_gamma=2.0,
        pos_weight=None,
    ):
        if loss not in {"binary_cross_entropy", "focal"}:
            raise ValueError(f"Unsupported loss: {loss}")
        if focal_gamma < 0.0:
            raise ValueError("focal_gamma must be >= 0")
        self.target = target
        self.loss_name = loss
        self.metrics = metrics or []
        self.focal_gamma = float(focal_gamma)
        self.pos_weight = normalize_pos_weight(pos_weight)

    def loss(self, batch, logits, stage):
        logits = self.predictions_for_metrics(batch, logits, stage=stage)
        targets = self.targets(batch, stage=stage).to(dtype=logits.dtype)
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        if self.loss_name == "focal":
            return focal_binary_cross_entropy_with_logits(
                logits,
                targets,
                gamma=self.focal_gamma,
                pos_weight=pos_weight,
            )
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    def predictions_for_metrics(self, batch, predictions, stage):
        del batch, stage
        if predictions.ndim == 2 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)
        if predictions.ndim != 1:
            raise ValueError("LinkPredictionTask expects one logit per candidate edge")
        return predictions

    def targets(self, batch, stage):
        del stage
        return batch.labels
