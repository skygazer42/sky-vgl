import torch

from vgl.tasks.base import Task


class SymmetricCrossEntropyTask(Task):
    def __init__(self, base_task, alpha=1.0, beta=1.0, label_clip=1e-4):
        if alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        if beta < 0.0:
            raise ValueError("beta must be >= 0")
        if label_clip <= 0.0 or label_clip > 1.0:
            raise ValueError("label_clip must be in (0.0, 1.0]")
        self.base_task = base_task
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.label_clip = float(label_clip)
        paired_loss = getattr(base_task, "paired_loss", None)
        if callable(paired_loss):
            self.paired_loss = self._paired_loss

    def __getattr__(self, name):
        return getattr(self.base_task, name)

    def _reverse_cross_entropy(self, batch, predictions, stage):
        logits = self.base_task.predictions_for_metrics(batch, predictions, stage)
        targets = self.base_task.targets(batch, stage).to(device=logits.device)
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
        elif logits.ndim == 2:
            probs = torch.softmax(logits, dim=-1)
            target_probs = torch.full_like(probs, self.label_clip)
            target_probs.scatter_(1, targets.to(dtype=torch.long).view(-1, 1), 1.0)
        else:
            raise ValueError("SymmetricCrossEntropyTask expects binary or multiclass logits")
        target_probs = target_probs.clamp_min(self.label_clip)
        return -(probs * torch.log(target_probs)).sum(dim=-1).mean()

    def _symmetric_cross_entropy(self, batch, predictions, stage):
        ce = self.base_task.loss(batch, predictions, stage)
        rce = self._reverse_cross_entropy(batch, predictions, stage)
        return self.alpha * ce + self.beta * rce

    def loss(self, batch, predictions, stage):
        if stage != "train":
            return self.base_task.loss(batch, predictions, stage)
        return self._symmetric_cross_entropy(batch, predictions, stage)

    def _paired_loss(self, batch, predictions, second_predictions, stage):
        if stage != "train":
            return self.base_task.paired_loss(batch, predictions, second_predictions, stage)
        base_paired = self.base_task.paired_loss(batch, predictions, second_predictions, stage)
        base_supervised = 0.5 * (
            self.base_task.loss(batch, predictions, stage)
            + self.base_task.loss(batch, second_predictions, stage)
        )
        sce = 0.5 * (
            self._symmetric_cross_entropy(batch, predictions, stage)
            + self._symmetric_cross_entropy(batch, second_predictions, stage)
        )
        return (base_paired - base_supervised) + sce

    def targets(self, batch, stage):
        return self.base_task.targets(batch, stage)

    def predictions_for_metrics(self, batch, predictions, stage):
        return self.base_task.predictions_for_metrics(batch, predictions, stage)


__all__ = ["SymmetricCrossEntropyTask"]
