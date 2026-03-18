import torch

from vgl.tasks.base import Task


class Poly1CrossEntropyTask(Task):
    def __init__(self, base_task, epsilon=1.0):
        if epsilon < 0.0:
            raise ValueError("epsilon must be >= 0")
        self.base_task = base_task
        self.epsilon = float(epsilon)
        paired_loss = getattr(base_task, "paired_loss", None)
        if callable(paired_loss):
            self.paired_loss = self._paired_loss

    def __getattr__(self, name):
        return getattr(self.base_task, name)

    def _true_class_probabilities(self, batch, predictions, stage):
        logits = self.base_task.predictions_for_metrics(batch, predictions, stage)
        targets = self.base_task.targets(batch, stage).to(device=logits.device)
        if logits.ndim == 1:
            probs = torch.sigmoid(logits)
            return torch.where(targets.to(dtype=logits.dtype) > 0.5, probs, 1.0 - probs)
        if logits.ndim == 2:
            probs = torch.softmax(logits, dim=-1)
            index = targets.to(dtype=torch.long).view(-1, 1)
            return probs.gather(-1, index).squeeze(-1)
        raise ValueError("Poly1CrossEntropyTask expects binary or multiclass logits")

    def _poly1_cross_entropy(self, batch, predictions, stage):
        supervised = self.base_task.loss(batch, predictions, stage)
        true_probs = self._true_class_probabilities(batch, predictions, stage)
        return supervised + self.epsilon * (1.0 - true_probs).mean()

    def loss(self, batch, predictions, stage):
        if stage != "train":
            return self.base_task.loss(batch, predictions, stage)
        return self._poly1_cross_entropy(batch, predictions, stage)

    def _paired_loss(self, batch, predictions, second_predictions, stage):
        if stage != "train":
            return self.base_task.paired_loss(batch, predictions, second_predictions, stage)
        base_paired = self.base_task.paired_loss(batch, predictions, second_predictions, stage)
        base_supervised = 0.5 * (
            self.base_task.loss(batch, predictions, stage)
            + self.base_task.loss(batch, second_predictions, stage)
        )
        poly1 = 0.5 * (
            self._poly1_cross_entropy(batch, predictions, stage)
            + self._poly1_cross_entropy(batch, second_predictions, stage)
        )
        return (base_paired - base_supervised) + poly1

    def targets(self, batch, stage):
        return self.base_task.targets(batch, stage)

    def predictions_for_metrics(self, batch, predictions, stage):
        return self.base_task.predictions_for_metrics(batch, predictions, stage)


__all__ = ["Poly1CrossEntropyTask"]
