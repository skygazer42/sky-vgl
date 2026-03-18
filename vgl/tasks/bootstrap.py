import torch
import torch.nn.functional as F

from vgl.tasks.base import Task


class BootstrapTask(Task):
    def __init__(self, base_task, beta=0.95, mode="soft"):
        if beta < 0.0 or beta > 1.0:
            raise ValueError("beta must be in [0.0, 1.0]")
        if mode not in {"soft", "hard"}:
            raise ValueError("mode must be 'soft' or 'hard'")
        self.base_task = base_task
        self.beta = float(beta)
        self.mode = mode
        paired_loss = getattr(base_task, "paired_loss", None)
        if callable(paired_loss):
            self.paired_loss = self._paired_loss

    def __getattr__(self, name):
        return getattr(self.base_task, name)

    def _bootstrap_loss(self, batch, predictions, stage):
        logits = self.base_task.predictions_for_metrics(batch, predictions, stage)
        targets = self.base_task.targets(batch, stage).to(device=logits.device)
        if logits.ndim == 1:
            probs = torch.sigmoid(logits).detach()
            pseudo_targets = probs if self.mode == "soft" else (probs >= 0.5).to(dtype=logits.dtype)
            blended_targets = self.beta * targets.to(dtype=logits.dtype) + (1.0 - self.beta) * pseudo_targets
            return F.binary_cross_entropy_with_logits(logits, blended_targets)
        if logits.ndim == 2:
            probs = torch.softmax(logits, dim=-1).detach()
            observed_targets = torch.zeros_like(probs)
            observed_targets.scatter_(1, targets.to(dtype=torch.long).view(-1, 1), 1.0)
            if self.mode == "soft":
                pseudo_targets = probs
            else:
                pseudo_targets = torch.zeros_like(probs)
                pseudo_targets.scatter_(1, probs.argmax(dim=-1, keepdim=True), 1.0)
            blended_targets = self.beta * observed_targets + (1.0 - self.beta) * pseudo_targets
            return -(blended_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        raise ValueError("BootstrapTask expects binary or multiclass logits")

    def loss(self, batch, predictions, stage):
        if stage != "train":
            return self.base_task.loss(batch, predictions, stage)
        return self._bootstrap_loss(batch, predictions, stage)

    def _paired_loss(self, batch, predictions, second_predictions, stage):
        if stage != "train":
            return self.base_task.paired_loss(batch, predictions, second_predictions, stage)
        base_paired = self.base_task.paired_loss(batch, predictions, second_predictions, stage)
        base_supervised = 0.5 * (
            self.base_task.loss(batch, predictions, stage)
            + self.base_task.loss(batch, second_predictions, stage)
        )
        bootstrap = 0.5 * (
            self._bootstrap_loss(batch, predictions, stage)
            + self._bootstrap_loss(batch, second_predictions, stage)
        )
        return (base_paired - base_supervised) + bootstrap

    def targets(self, batch, stage):
        return self.base_task.targets(batch, stage)

    def predictions_for_metrics(self, batch, predictions, stage):
        return self.base_task.predictions_for_metrics(batch, predictions, stage)


__all__ = ["BootstrapTask"]
