import torch

from vgl.tasks.base import Task


class ConfidencePenaltyTask(Task):
    def __init__(self, base_task, coefficient=0.0):
        if coefficient < 0.0:
            raise ValueError("coefficient must be >= 0")
        self.base_task = base_task
        self.coefficient = float(coefficient)
        paired_loss = getattr(base_task, "paired_loss", None)
        if callable(paired_loss):
            self.paired_loss = self._paired_loss

    def __getattr__(self, name):
        return getattr(self.base_task, name)

    def _entropy(self, predictions):
        if predictions.ndim == 1:
            probs = torch.sigmoid(predictions)
            return -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs)).mean()
        probs = torch.softmax(predictions, dim=-1)
        return -(probs * torch.log(probs)).sum(dim=-1).mean()

    def _apply_penalty(self, loss, predictions, stage):
        if stage != "train" or self.coefficient == 0.0:
            return loss
        return loss - self.coefficient * self._entropy(predictions)

    def loss(self, batch, predictions, stage):
        return self._apply_penalty(
            self.base_task.loss(batch, predictions, stage),
            self.base_task.predictions_for_metrics(batch, predictions, stage),
            stage,
        )

    def _paired_loss(self, batch, predictions, second_predictions, stage):
        loss = self.base_task.paired_loss(batch, predictions, second_predictions, stage)
        if stage != "train" or self.coefficient == 0.0:
            return loss
        primary = self.base_task.predictions_for_metrics(batch, predictions, stage)
        secondary = self.base_task.predictions_for_metrics(batch, second_predictions, stage)
        entropy = 0.5 * (self._entropy(primary) + self._entropy(secondary))
        return loss - self.coefficient * entropy

    def targets(self, batch, stage):
        return self.base_task.targets(batch, stage)

    def predictions_for_metrics(self, batch, predictions, stage):
        return self.base_task.predictions_for_metrics(batch, predictions, stage)


__all__ = ["ConfidencePenaltyTask"]
