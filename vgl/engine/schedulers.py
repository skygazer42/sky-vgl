import math

import torch


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_ratio=0.0):
        resolved_warmup_epochs = _as_python_int(warmup_epochs)
        resolved_max_epochs = _as_python_int(max_epochs)
        if resolved_warmup_epochs < 1:
            raise ValueError("warmup_epochs must be >= 1")
        if resolved_max_epochs <= resolved_warmup_epochs:
            raise ValueError("max_epochs must be > warmup_epochs")
        if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

        self.optimizer = optimizer
        self.warmup_epochs = resolved_warmup_epochs
        self.max_epochs = resolved_max_epochs
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.completed_epochs = 0
        self._last_lr = []
        self._apply_current_lr()

    def _factor_for_completed_epochs(self, completed_epochs):
        if completed_epochs < self.warmup_epochs:
            return float(completed_epochs + 1) / float(self.warmup_epochs)

        decay_epochs = self.max_epochs - self.warmup_epochs
        decay_step = min(max(completed_epochs - self.warmup_epochs + 1, 1), decay_epochs)
        progress = float(decay_step) / float(decay_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply_current_lr(self):
        factor = self._factor_for_completed_epochs(self.completed_epochs)
        self._last_lr = []
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = base_lr * factor
            group["lr"] = lr
            self._last_lr.append(lr)

    def step(self):
        self.completed_epochs += 1
        self._apply_current_lr()
        return self.get_last_lr()

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return {
            "warmup_epochs": self.warmup_epochs,
            "max_epochs": self.max_epochs,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "completed_epochs": self.completed_epochs,
        }

    def load_state_dict(self, state):
        self.base_lrs = [float(lr) for lr in state.get("base_lrs", self.base_lrs)]
        self.completed_epochs = _as_python_int(state.get("completed_epochs", 0))
        self._apply_current_lr()


__all__ = ["WarmupCosineScheduler"]
