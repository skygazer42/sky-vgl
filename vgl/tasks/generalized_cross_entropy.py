import torch

from vgl.tasks.base import Task


class GeneralizedCrossEntropyTask(Task):
    def __init__(self, base_task, q=0.7):
        if q <= 0.0 or q > 1.0:
            raise ValueError("q must be in (0.0, 1.0]")
        self.base_task = base_task
        self.q = float(q)
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
        raise ValueError("GeneralizedCrossEntropyTask expects binary or multiclass logits")

    def _generalized_cross_entropy(self, batch, predictions, stage):
        true_probs = self._true_class_probabilities(batch, predictions, stage)
        return ((1.0 - true_probs.pow(self.q)) / self.q).mean()

    def loss(self, batch, predictions, stage):
        if stage != "train":
            return self.base_task.loss(batch, predictions, stage)
        return self._generalized_cross_entropy(batch, predictions, stage)

    def _paired_loss(self, batch, predictions, second_predictions, stage):
        if stage != "train":
            return self.base_task.paired_loss(batch, predictions, second_predictions, stage)
        base_paired = self.base_task.paired_loss(batch, predictions, second_predictions, stage)
        base_supervised = 0.5 * (
            self.base_task.loss(batch, predictions, stage)
            + self.base_task.loss(batch, second_predictions, stage)
        )
        gce = 0.5 * (
            self._generalized_cross_entropy(batch, predictions, stage)
            + self._generalized_cross_entropy(batch, second_predictions, stage)
        )
        return (base_paired - base_supervised) + gce

    def targets(self, batch, stage):
        return self.base_task.targets(batch, stage)

    def predictions_for_metrics(self, batch, predictions, stage):
        return self.base_task.predictions_for_metrics(batch, predictions, stage)


__all__ = ["GeneralizedCrossEntropyTask"]
