import torch.nn.functional as F

from vgl.tasks.base import Task


class RDropTask(Task):
    def __init__(self, base_task, alpha=1.0):
        if alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        if getattr(base_task, "loss_name", None) != "cross_entropy":
            raise ValueError("RDropTask requires a cross_entropy base task")
        self.base_task = base_task
        self.alpha = float(alpha)

    def __getattr__(self, name):
        return getattr(self.base_task, name)

    def loss(self, batch, predictions, stage):
        return self.base_task.loss(batch, predictions, stage)

    def paired_loss(self, batch, predictions, second_predictions, stage):
        primary_loss = self.base_task.loss(batch, predictions, stage)
        secondary_loss = self.base_task.loss(batch, second_predictions, stage)
        primary_logits = self.base_task.predictions_for_metrics(batch, predictions, stage)
        secondary_logits = self.base_task.predictions_for_metrics(batch, second_predictions, stage)
        return 0.5 * (primary_loss + secondary_loss) + self.alpha * self._symmetric_kl(
            primary_logits,
            secondary_logits,
        )

    def targets(self, batch, stage):
        return self.base_task.targets(batch, stage)

    def predictions_for_metrics(self, batch, predictions, stage):
        return self.base_task.predictions_for_metrics(batch, predictions, stage)

    def _symmetric_kl(self, predictions, second_predictions):
        kl_ab = F.kl_div(
            F.log_softmax(predictions, dim=-1),
            F.softmax(second_predictions, dim=-1),
            reduction="batchmean",
        )
        kl_ba = F.kl_div(
            F.log_softmax(second_predictions, dim=-1),
            F.softmax(predictions, dim=-1),
            reduction="batchmean",
        )
        return 0.5 * (kl_ab + kl_ba)


__all__ = ["RDropTask"]
