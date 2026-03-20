from pathlib import Path
from time import perf_counter

import torch

from vgl.engine.checkpoints import checkpoint_event_fields
from vgl.engine.monitoring import extract_monitor_value, is_improvement, resolve_monitor_mode


class Callback:
    def on_fit_start(self, trainer, history):
        del trainer, history

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        del state

    def on_before_optimizer_step(self, trainer, step):
        del trainer, step

    def on_after_optimizer_step(self, trainer, step):
        del trainer, step

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, epoch, train_summary, val_summary, history

    def on_fit_end(self, trainer, history):
        del trainer, history

    def on_exception(self, trainer, exception, history):
        del trainer, exception, history


class StopTraining(Exception):
    pass


class EarlyStopping(Callback):
    def __init__(self, patience=0, monitor=None, mode=None, min_delta=0.0):
        if patience < 0:
            raise ValueError("patience must be >= 0")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        if mode not in {None, "min", "max"}:
            raise ValueError("mode must be 'min', 'max', or None")
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best_value = None
        self.bad_epochs = 0

    def on_fit_start(self, trainer, history):
        del trainer, history
        self.best_value = None
        self.bad_epochs = 0

    def state_dict(self):
        return {
            "best_value": self.best_value,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state):
        self.best_value = state.get("best_value")
        self.bad_epochs = int(state.get("bad_epochs", 0))

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, epoch
        monitor = self.monitor or history["monitor"]
        mode = resolve_monitor_mode(
            monitor,
            mode=self.mode,
            invalid_mode_message="mode must be 'min' or 'max'",
        )
        current = extract_monitor_value(
            monitor,
            train_summary=train_summary,
            val_summary=val_summary,
            error_subject="Callback monitor",
        )
        improved = is_improvement(
            current,
            self.best_value,
            mode,
            min_delta=self.min_delta,
        )
        if improved:
            self.best_value = current
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            raise StopTraining(f"Early stopping on {monitor}")


class HistoryLogger(Callback):
    def __init__(self, sink=None):
        self.sink = sink
        self.records = []

    def on_fit_start(self, trainer, history):
        del trainer, history
        self.records = []

    def state_dict(self):
        return {
            "records": [dict(record) for record in self.records],
        }

    def load_state_dict(self, state):
        self.records = [dict(record) for record in state.get("records", [])]

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        elapsed_seconds = None
        if history["epoch_elapsed_seconds"]:
            elapsed_seconds = history["epoch_elapsed_seconds"][-1]
        record = {
            "epoch": epoch,
            "train": dict(train_summary),
            "val": None if val_summary is None else dict(val_summary),
            "best_epoch": history["best_epoch"],
            "best_metric": history["best_metric"],
            "monitor": history["monitor"],
            "global_step": trainer.global_step,
            "elapsed_seconds": elapsed_seconds,
        }
        self.records.append(record)
        if self.sink is not None:
            self.sink(record)


def _unitwise_norm(tensor):
    if tensor.ndim <= 1:
        return tensor.abs()
    dims = tuple(range(1, tensor.ndim))
    return torch.linalg.vector_norm(tensor, dim=dims, keepdim=True)


class AdaptiveGradientClipping(Callback):
    def __init__(self, clipping=0.01, eps=1e-3):
        if clipping <= 0.0:
            raise ValueError("clipping must be > 0")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        self.clipping = float(clipping)
        self.eps = float(eps)

    def on_before_optimizer_step(self, trainer, step):
        del step
        for parameter in trainer.model.parameters():
            grad = parameter.grad
            if grad is None or grad.is_sparse:
                continue
            param_norm = _unitwise_norm(parameter.detach()).clamp(min=self.eps)
            grad_norm = _unitwise_norm(grad.detach())
            max_norm = param_norm * self.clipping
            trigger = grad_norm > max_norm
            clipped_grad = grad * (max_norm / grad_norm.clamp(min=1e-6))
            grad.copy_(torch.where(trigger, clipped_grad, grad))


class GradientValueClipping(Callback):
    def __init__(self, clip_value):
        if clip_value <= 0.0:
            raise ValueError("clip_value must be > 0")
        self.clip_value = float(clip_value)

    def on_before_optimizer_step(self, trainer, step):
        del step
        for parameter in trainer.model.parameters():
            grad = parameter.grad
            if grad is None or grad.is_sparse:
                continue
            grad.clamp_(min=-self.clip_value, max=self.clip_value)


class GradientCentralization(Callback):
    def __init__(self, conv_only=False):
        if not isinstance(conv_only, bool):
            raise TypeError("conv_only must be a bool")
        self.conv_only = conv_only

    def on_before_optimizer_step(self, trainer, step):
        del step
        min_rank = 4 if self.conv_only else 2
        for parameter in trainer.model.parameters():
            grad = parameter.grad
            if grad is None or grad.is_sparse or grad.ndim < min_rank:
                continue
            dims = tuple(range(1, grad.ndim))
            grad.sub_(grad.mean(dim=dims, keepdim=True))


class GradientNoiseInjection(Callback):
    def __init__(self, std, decay_exponent=0.0, seed=0):
        if std <= 0.0:
            raise ValueError("std must be > 0")
        if decay_exponent < 0.0:
            raise ValueError("decay_exponent must be >= 0")
        self.std = float(std)
        self.decay_exponent = float(decay_exponent)
        self.seed = int(seed)
        self.step_count = 0
        self._generator = None

    def _ensure_generator(self):
        if self._generator is None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(self.seed)
        return self._generator

    def _std_for_step(self, step):
        step = int(step)
        return self.std / (float(step) ** self.decay_exponent)

    def on_fit_start(self, trainer, history):
        del trainer, history
        self.step_count = 0
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

    def state_dict(self):
        generator_state = None
        if self._generator is not None:
            generator_state = self._generator.get_state()
        return {
            "step_count": self.step_count,
            "generator_state": generator_state,
        }

    def load_state_dict(self, state):
        self.step_count = int(state.get("step_count", 0))
        generator_state = state.get("generator_state")
        if generator_state is None:
            return
        self._ensure_generator().set_state(generator_state)

    def on_before_optimizer_step(self, trainer, step):
        current_std = self._std_for_step(step)
        generator = self._ensure_generator()
        for parameter in trainer.model.parameters():
            grad = parameter.grad
            if grad is None or grad.is_sparse:
                continue
            noise = torch.randn(
                grad.shape,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            ).to(device=grad.device, dtype=grad.dtype)
            grad.add_(noise, alpha=current_std)
        self.step_count = int(step)


class DeferredReweighting(Callback):
    def __init__(self, start_epoch=2, beta=0.9999):
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if beta < 0.0 or beta >= 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.start_epoch = int(start_epoch)
        self.beta = float(beta)
        self.original_class_weight = None
        self.reweighting_active = False
        self._task = None

    def _resolve_task(self, task):
        while hasattr(task, "base_task"):
            task = task.base_task
        return task

    def _validate_task(self, task):
        if not hasattr(task, "class_count"):
            raise ValueError("DeferredReweighting requires task.class_count")
        if task.class_count is None:
            raise ValueError("DeferredReweighting requires class_count")

    def _drw_weight(self):
        count = self._task.class_count.to(dtype=torch.float32)
        effective_num = 1.0 - torch.pow(torch.full_like(count, self.beta), count)
        weight = (1.0 - self.beta) / effective_num.clamp(min=1e-12)
        return weight / weight.sum() * weight.numel()

    def _apply_class_weight(self):
        if self._task is None:
            return
        if not self.reweighting_active:
            if self.original_class_weight is None:
                self._task.class_weight = None
            else:
                self._task.class_weight = self.original_class_weight.detach().clone()
            return
        self._task.class_weight = self._drw_weight()

    def on_fit_start(self, trainer, history):
        del history
        self._task = self._resolve_task(trainer.task)
        self._validate_task(self._task)
        if self._task.class_weight is None:
            self.original_class_weight = None
        else:
            self.original_class_weight = self._task.class_weight.detach().clone()
        self.reweighting_active = self.start_epoch <= 1
        self._apply_class_weight()

    def state_dict(self):
        return {
            "reweighting_active": self.reweighting_active,
        }

    def load_state_dict(self, state):
        self.reweighting_active = bool(state.get("reweighting_active", self.reweighting_active))
        self._apply_class_weight()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        next_active = epoch + 1 >= self.start_epoch
        if next_active == self.reweighting_active:
            return
        self.reweighting_active = next_active
        self._apply_class_weight()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is None:
            return
        if self.original_class_weight is None:
            self._task.class_weight = None
        else:
            self._task.class_weight = self.original_class_weight.detach().clone()


class LabelSmoothingScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value < 0.0 or start_value >= 1.0:
            raise ValueError("start_value must be in [0, 1)")
        if end_value < 0.0 or end_value >= 1.0:
            raise ValueError("end_value must be in [0, 1)")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_label_smoothing = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while hasattr(task, "base_task"):
            task = task.base_task
        return task

    def _validate_task(self, task):
        if not hasattr(task, "label_smoothing"):
            raise ValueError("LabelSmoothingScheduler requires task.label_smoothing")

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_label_smoothing(self):
        if self._task is not None:
            self._task.label_smoothing = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self._validate_task(self._task)
        self.original_label_smoothing = float(self._task.label_smoothing)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_label_smoothing()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_label_smoothing()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_label_smoothing()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_label_smoothing is not None:
            self._task.label_smoothing = float(self.original_label_smoothing)


class FocalGammaScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value < 0.0:
            raise ValueError("start_value must be >= 0")
        if end_value < 0.0:
            raise ValueError("end_value must be >= 0")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_focal_gamma = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while hasattr(task, "base_task"):
            task = task.base_task
        return task

    def _validate_task(self, task):
        if not hasattr(task, "focal_gamma"):
            raise ValueError("FocalGammaScheduler requires task.focal_gamma")

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_focal_gamma(self):
        if self._task is not None:
            self._task.focal_gamma = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self._validate_task(self._task)
        self.original_focal_gamma = float(self._task.focal_gamma)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_focal_gamma()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_focal_gamma()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_focal_gamma()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_focal_gamma is not None:
            self._task.focal_gamma = float(self.original_focal_gamma)


class LogitAdjustTauScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value < 0.0:
            raise ValueError("start_value must be >= 0")
        if end_value < 0.0:
            raise ValueError("end_value must be >= 0")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_logit_adjust_tau = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while hasattr(task, "base_task"):
            task = task.base_task
        return task

    def _validate_task(self, task):
        if not hasattr(task, "logit_adjust_tau"):
            raise ValueError("LogitAdjustTauScheduler requires task.logit_adjust_tau")

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_logit_adjust_tau(self):
        if self._task is not None:
            self._task.logit_adjust_tau = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self._validate_task(self._task)
        self.original_logit_adjust_tau = float(self._task.logit_adjust_tau)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_logit_adjust_tau()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_logit_adjust_tau()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_logit_adjust_tau()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_logit_adjust_tau is not None:
            self._task.logit_adjust_tau = float(self.original_logit_adjust_tau)


class LdamMarginScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value <= 0.0:
            raise ValueError("start_value must be > 0")
        if end_value <= 0.0:
            raise ValueError("end_value must be > 0")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_ldam_max_margin = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while hasattr(task, "base_task"):
            task = task.base_task
        return task

    def _validate_task(self, task):
        if not hasattr(task, "ldam_max_margin"):
            raise ValueError("LdamMarginScheduler requires task.ldam_max_margin")

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_ldam_max_margin(self):
        if self._task is not None:
            self._task.ldam_max_margin = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self._validate_task(self._task)
        self.original_ldam_max_margin = float(self._task.ldam_max_margin)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_ldam_max_margin()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_ldam_max_margin()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_ldam_max_margin()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_ldam_max_margin is not None:
            self._task.ldam_max_margin = float(self.original_ldam_max_margin)


class PosWeightScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value <= 0.0:
            raise ValueError("start_value must be > 0")
        if end_value <= 0.0:
            raise ValueError("end_value must be > 0")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_pos_weight = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while hasattr(task, "base_task"):
            task = task.base_task
        return task

    def _validate_task(self, task):
        if not hasattr(task, "pos_weight"):
            raise ValueError("PosWeightScheduler requires task.pos_weight")

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _scheduled_pos_weight(self):
        if self.original_pos_weight is None:
            return torch.tensor([self.current_value], dtype=torch.float32)
        return torch.full_like(self.original_pos_weight, self.current_value)

    def _apply_pos_weight(self):
        if self._task is not None:
            self._task.pos_weight = self._scheduled_pos_weight()

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self._validate_task(self._task)
        if self._task.pos_weight is None:
            self.original_pos_weight = None
        else:
            self.original_pos_weight = self._task.pos_weight.detach().clone()
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_pos_weight()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_pos_weight()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_pos_weight()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is None:
            return
        if self.original_pos_weight is None:
            self._task.pos_weight = None
        else:
            self._task.pos_weight = self.original_pos_weight.detach().clone()


class BootstrapBetaScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value < 0.0 or start_value > 1.0:
            raise ValueError("start_value must be in [0, 1]")
        if end_value < 0.0 or end_value > 1.0:
            raise ValueError("end_value must be in [0, 1]")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_beta = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while not hasattr(task, "beta"):
            if not hasattr(task, "base_task"):
                raise ValueError("BootstrapBetaScheduler requires task.beta")
            task = task.base_task
        return task

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_beta(self):
        if self._task is not None:
            self._task.beta = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self.original_beta = float(self._task.beta)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_beta()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_beta()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_beta()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_beta is not None:
            self._task.beta = float(self.original_beta)


class ConfidencePenaltyScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value < 0.0:
            raise ValueError("start_value must be >= 0")
        if end_value < 0.0:
            raise ValueError("end_value must be >= 0")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_coefficient = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while not hasattr(task, "coefficient"):
            if not hasattr(task, "base_task"):
                raise ValueError("ConfidencePenaltyScheduler requires task.coefficient")
            task = task.base_task
        return task

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_coefficient(self):
        if self._task is not None:
            self._task.coefficient = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self.original_coefficient = float(self._task.coefficient)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_coefficient()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_coefficient()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_coefficient()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_coefficient is not None:
            self._task.coefficient = float(self.original_coefficient)


class FloodingLevelScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value < 0.0:
            raise ValueError("start_value must be >= 0")
        if end_value < 0.0:
            raise ValueError("end_value must be >= 0")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_level = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while not hasattr(task, "level"):
            if not hasattr(task, "base_task"):
                raise ValueError("FloodingLevelScheduler requires task.level")
            task = task.base_task
        return task

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_level(self):
        if self._task is not None:
            self._task.level = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self.original_level = float(self._task.level)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_level()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_level()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_level()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_level is not None:
            self._task.level = float(self.original_level)


class GeneralizedCrossEntropyScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value <= 0.0 or start_value > 1.0:
            raise ValueError("start_value must be in (0, 1]")
        if end_value <= 0.0 or end_value > 1.0:
            raise ValueError("end_value must be in (0, 1]")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_q = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while not hasattr(task, "q"):
            if not hasattr(task, "base_task"):
                raise ValueError("GeneralizedCrossEntropyScheduler requires task.q")
            task = task.base_task
        return task

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_q(self):
        if self._task is not None:
            self._task.q = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self.original_q = float(self._task.q)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_q()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_q()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_q()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_q is not None:
            self._task.q = float(self.original_q)


class Poly1EpsilonScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value < 0.0:
            raise ValueError("start_value must be >= 0")
        if end_value < 0.0:
            raise ValueError("end_value must be >= 0")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_epsilon = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while not hasattr(task, "epsilon"):
            if not hasattr(task, "base_task"):
                raise ValueError("Poly1EpsilonScheduler requires task.epsilon")
            task = task.base_task
        return task

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_epsilon(self):
        if self._task is not None:
            self._task.epsilon = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self.original_epsilon = float(self._task.epsilon)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_epsilon()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_epsilon()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_epsilon()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_epsilon is not None:
            self._task.epsilon = float(self.original_epsilon)


class SymmetricCrossEntropyBetaScheduler(Callback):
    def __init__(self, start_value, end_value, *, start_epoch=1, end_epoch=1):
        if start_value < 0.0:
            raise ValueError("start_value must be >= 0")
        if end_value < 0.0:
            raise ValueError("end_value must be >= 0")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_beta = None
        self.current_value = float(start_value)
        self._task = None

    def _resolve_task(self, task):
        while not (
            hasattr(task, "beta") and hasattr(task, "alpha") and hasattr(task, "label_clip")
        ):
            if not hasattr(task, "base_task"):
                raise ValueError(
                    "SymmetricCrossEntropyBetaScheduler requires SymmetricCrossEntropyTask.beta"
                )
            task = task.base_task
        return task

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_value
        if self.end_epoch == self.start_epoch:
            return self.end_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_beta(self):
        if self._task is not None:
            self._task.beta = float(self.current_value)

    def on_fit_start(self, trainer, history):
        self._task = self._resolve_task(trainer.task)
        self.original_beta = float(self._task.beta)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_value = self._value_for_epoch(next_epoch)
        self._apply_beta()

    def state_dict(self):
        return {
            "current_value": self.current_value,
        }

    def load_state_dict(self, state):
        self.current_value = float(state.get("current_value", self.current_value))
        self._apply_beta()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_value = self._value_for_epoch(epoch + 1)
        self._apply_beta()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._task is not None and self.original_beta is not None:
            self._task.beta = float(self.original_beta)


class GradientAccumulationScheduler(Callback):
    def __init__(self, scheduling):
        self.scheduling = self._normalize_scheduling(scheduling)
        self.original_accumulate_grad_batches = None
        self.current_accumulate_grad_batches = None
        self._trainer = None

    def _normalize_scheduling(self, scheduling):
        if isinstance(scheduling, dict):
            items = list(scheduling.items())
        else:
            items = list(scheduling)
        if not items:
            raise ValueError("scheduling must not be empty")

        normalized = {}
        for epoch, accumulate_grad_batches in items:
            if not isinstance(epoch, int) or isinstance(epoch, bool):
                raise TypeError("scheduling epoch keys must be integers")
            if epoch < 1:
                raise ValueError("scheduling epoch keys must be >= 1")
            if not isinstance(accumulate_grad_batches, int) or isinstance(accumulate_grad_batches, bool):
                raise TypeError("scheduling accumulation values must be integers")
            if accumulate_grad_batches < 1:
                raise ValueError("scheduling accumulation values must be >= 1")
            normalized[int(epoch)] = int(accumulate_grad_batches)
        return tuple(sorted(normalized.items()))

    def _value_for_epoch(self, epoch):
        if self.original_accumulate_grad_batches is None:
            raise RuntimeError("GradientAccumulationScheduler is not initialized")
        value = int(self.original_accumulate_grad_batches)
        for start_epoch, scheduled_accumulation in self.scheduling:
            if epoch < start_epoch:
                break
            value = int(scheduled_accumulation)
        return value

    def _apply_to_trainer(self):
        if self._trainer is None:
            return
        if self.current_accumulate_grad_batches is None:
            return
        self._trainer.accumulate_grad_batches = int(self.current_accumulate_grad_batches)

    def on_fit_start(self, trainer, history):
        self._trainer = trainer
        self.original_accumulate_grad_batches = int(trainer.accumulate_grad_batches)
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_accumulate_grad_batches = self._value_for_epoch(next_epoch)
        self._apply_to_trainer()

    def state_dict(self):
        return {
            "original_accumulate_grad_batches": self.original_accumulate_grad_batches,
            "current_accumulate_grad_batches": self.current_accumulate_grad_batches,
        }

    def load_state_dict(self, state):
        original = state.get("original_accumulate_grad_batches")
        if original is not None:
            self.original_accumulate_grad_batches = int(original)
        current = state.get("current_accumulate_grad_batches")
        if current is not None:
            self.current_accumulate_grad_batches = int(current)
        self._apply_to_trainer()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_accumulate_grad_batches = self._value_for_epoch(epoch + 1)
        self._apply_to_trainer()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._trainer is None or self.original_accumulate_grad_batches is None:
            return
        self._trainer.accumulate_grad_batches = int(self.original_accumulate_grad_batches)


class ModelCheckpoint(Callback):
    def __init__(
        self,
        dirpath,
        *,
        filename="epoch{epoch}",
        monitor=None,
        mode=None,
        save_top_k=1,
        save_last=True,
        save_on_exception=False,
        every_n_epochs=1,
    ):
        if not isinstance(filename, str) or not filename:
            raise ValueError("filename must be a non-empty string")
        if mode not in {None, "min", "max"}:
            raise ValueError("mode must be 'min', 'max', or None")
        if not isinstance(save_top_k, int) or isinstance(save_top_k, bool):
            raise TypeError("save_top_k must be an integer")
        if save_top_k < -1:
            raise ValueError("save_top_k must be >= -1")
        if not isinstance(save_last, bool):
            raise TypeError("save_last must be a bool")
        if not isinstance(save_on_exception, bool):
            raise TypeError("save_on_exception must be a bool")
        if not isinstance(every_n_epochs, int) or isinstance(every_n_epochs, bool):
            raise TypeError("every_n_epochs must be an integer")
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be >= 1")

        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = int(save_top_k)
        self.save_last = bool(save_last)
        self.save_on_exception = bool(save_on_exception)
        self.every_n_epochs = int(every_n_epochs)
        self.active_monitor = None
        self.active_mode = None
        self.best_k_models = []
        self.best_model_path = None
        self.best_model_score = None
        self.kth_best_model_path = None
        self.kth_best_model_score = None
        self.last_model_path = None
        self.exception_model_path = None

    def _checkpointing_enabled(self, trainer):
        return bool(getattr(trainer, "_checkpointing_enabled", True))

    def _ranked_models(self):
        reverse = self.active_mode == "max"
        return sorted(self.best_k_models, key=lambda entry: entry["score"], reverse=reverse)

    def _refresh_best_model_fields(self):
        if not self.best_k_models:
            self.best_model_path = None
            self.best_model_score = None
            self.kth_best_model_path = None
            self.kth_best_model_score = None
            return
        ranked = self._ranked_models()
        self.best_model_path = ranked[0]["path"]
        self.best_model_score = float(ranked[0]["score"])
        self.kth_best_model_path = ranked[-1]["path"]
        self.kth_best_model_score = float(ranked[-1]["score"])

    def _worst_saved_score(self):
        if not self.best_k_models:
            return None
        ranked = self._ranked_models()
        return float(ranked[-1]["score"])

    def _format_checkpoint_path(self, *, epoch, monitor_value):
        format_values = {"epoch": int(epoch)}
        if monitor_value is not None:
            format_values["monitor"] = float(monitor_value)
        try:
            filename = self.filename.format(**format_values)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(f"filename template references unknown field '{missing}'") from exc
        if not filename.endswith(".ckpt"):
            filename = f"{filename}.ckpt"
        return self.dirpath / filename

    def _save_checkpoint(self, trainer, history, path, *, epoch, tag, monitor_value=None):
        metadata = {
            "epoch": int(epoch),
            "global_step": int(trainer.global_step),
            "tag": str(tag),
        }
        if self.active_monitor is not None and monitor_value is not None:
            metadata["monitor"] = self.active_monitor
            metadata["monitor_value"] = float(monitor_value)
        save_start_time = perf_counter()
        checkpoint_path = trainer.save_training_checkpoint(path, history=history, metadata=metadata)
        event_fields = checkpoint_event_fields(
            checkpoint_path,
            save_seconds=perf_counter() - save_start_time,
        )
        trainer.log_event(
            "checkpoint_saved",
            stage="fit",
            metrics=(
                None
                if self.active_monitor is None or monitor_value is None
                else {self.active_monitor: float(monitor_value)}
            ),
            checkpoint_tag=str(tag),
            monitor_name=self.active_monitor,
            monitor_value=None if monitor_value is None else float(monitor_value),
            **event_fields,
        )

    def _save_last_checkpoint(self, trainer, history, *, epoch):
        if not self.save_last:
            return
        path = self.dirpath / "last.ckpt"
        self._save_checkpoint(trainer, history, path, epoch=epoch, tag="last")
        self.last_model_path = str(path)

    def _save_exception_checkpoint(self, trainer, exception, history):
        if not self.save_on_exception:
            return
        path = self.dirpath / "exception.ckpt"
        metadata = {
            "epoch": int(history["completed_epochs"]),
            "global_step": int(trainer.global_step),
            "tag": "exception",
            "exception_type": exception.__class__.__name__,
            "exception_message": str(exception),
        }
        save_start_time = perf_counter()
        checkpoint_path = trainer.save_training_checkpoint(path, history=history, metadata=metadata)
        self.exception_model_path = str(checkpoint_path)
        event_fields = checkpoint_event_fields(
            checkpoint_path,
            save_seconds=perf_counter() - save_start_time,
        )
        trainer.log_event(
            "checkpoint_saved",
            stage="fit",
            checkpoint_tag="exception",
            exception_type=exception.__class__.__name__,
            exception_message=str(exception),
            **event_fields,
        )

    def _upsert_tracked_model(self, path, score):
        path_str = str(path)
        self.best_k_models = [entry for entry in self.best_k_models if entry["path"] != path_str]
        self.best_k_models.append({"path": path_str, "score": float(score)})

    def _trim_extra_models(self):
        if self.save_top_k < 0:
            return
        if len(self.best_k_models) <= self.save_top_k:
            return
        ranked = self._ranked_models()
        kept = ranked[: self.save_top_k]
        removed = ranked[self.save_top_k :]
        self.best_k_models = kept
        for entry in removed:
            remove_path = Path(entry["path"])
            if self.save_last and self.last_model_path == entry["path"]:
                continue
            if remove_path.exists():
                remove_path.unlink()

    def on_fit_start(self, trainer, history):
        if not self._checkpointing_enabled(trainer):
            self.active_monitor = None
            self.active_mode = None
            self.best_k_models = []
            self.best_model_path = None
            self.best_model_score = None
            self.kth_best_model_path = None
            self.kth_best_model_score = None
            self.last_model_path = None
            self.exception_model_path = None
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        if self.save_top_k != 0 or self.monitor is not None:
            self.active_monitor = self.monitor or history["monitor"]
            self.active_mode = resolve_monitor_mode(
                self.active_monitor,
                mode=self.mode,
                invalid_mode_message="mode must be 'min' or 'max'",
            )
        else:
            self.active_monitor = None
            self.active_mode = None
        self._refresh_best_model_fields()

    def state_dict(self):
        return {
            "active_monitor": self.active_monitor,
            "active_mode": self.active_mode,
            "best_k_models": [dict(entry) for entry in self.best_k_models],
            "best_model_path": self.best_model_path,
            "best_model_score": self.best_model_score,
            "kth_best_model_path": self.kth_best_model_path,
            "kth_best_model_score": self.kth_best_model_score,
            "last_model_path": self.last_model_path,
            "exception_model_path": self.exception_model_path,
        }

    def load_state_dict(self, state):
        state_monitor = state.get("active_monitor")
        state_mode = state.get("active_mode")
        if self.active_monitor is not None and state_monitor not in {None, self.active_monitor}:
            raise ValueError("ModelCheckpoint monitor does not match checkpoint state")
        if self.active_mode is not None and state_mode not in {None, self.active_mode}:
            raise ValueError("ModelCheckpoint mode does not match checkpoint state")
        if state_monitor is not None:
            self.active_monitor = state_monitor
        if state_mode is not None:
            self.active_mode = state_mode
        self.best_k_models = [
            {"path": str(entry["path"]), "score": float(entry["score"])}
            for entry in state.get("best_k_models", [])
        ]
        self.best_model_path = state.get("best_model_path")
        best_score = state.get("best_model_score")
        self.best_model_score = None if best_score is None else float(best_score)
        self.kth_best_model_path = state.get("kth_best_model_path")
        kth_score = state.get("kth_best_model_score")
        self.kth_best_model_score = None if kth_score is None else float(kth_score)
        self.last_model_path = state.get("last_model_path")
        self.exception_model_path = state.get("exception_model_path")
        self._refresh_best_model_fields()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        if not self._checkpointing_enabled(trainer):
            return
        if epoch % self.every_n_epochs != 0:
            return

        self._save_last_checkpoint(trainer, history, epoch=epoch)
        if self.save_top_k == 0:
            return
        if self.active_monitor is None or self.active_mode is None:
            raise RuntimeError("ModelCheckpoint monitor is not configured")

        monitor_value = extract_monitor_value(
            self.active_monitor,
            train_summary=train_summary,
            val_summary=val_summary,
            error_subject="ModelCheckpoint monitor",
        )
        should_save = self.save_top_k < 0 or len(self.best_k_models) < self.save_top_k
        if not should_save:
            worst_score = self._worst_saved_score()
            should_save = is_improvement(monitor_value, worst_score, self.active_mode)
        if not should_save:
            return

        model_path = self._format_checkpoint_path(epoch=epoch, monitor_value=monitor_value)
        self._save_checkpoint(
            trainer,
            history,
            model_path,
            epoch=epoch,
            tag="top_k",
            monitor_value=monitor_value,
        )
        self._upsert_tracked_model(model_path, monitor_value)
        self._trim_extra_models()
        self._refresh_best_model_fields()

    def on_exception(self, trainer, exception, history):
        if not self._checkpointing_enabled(trainer):
            return
        self._save_exception_checkpoint(trainer, exception, history)


class WeightDecayScheduler(Callback):
    def __init__(self, start_factor, end_factor, *, start_epoch=1, end_epoch=1):
        if start_factor < 0.0:
            raise ValueError("start_factor must be >= 0")
        if end_factor < 0.0:
            raise ValueError("end_factor must be >= 0")
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        self.start_factor = float(start_factor)
        self.end_factor = float(end_factor)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.original_weight_decays = []
        self.current_factor = float(start_factor)
        self._optimizer = None

    def _value_for_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.start_factor
        if self.end_epoch == self.start_epoch:
            return self.end_factor
        if epoch >= self.end_epoch:
            return self.end_factor
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_factor + progress * (self.end_factor - self.start_factor)

    def _apply_weight_decay(self):
        if self._optimizer is None:
            return
        for group, original_weight_decay in zip(self._optimizer.param_groups, self.original_weight_decays):
            group["weight_decay"] = float(original_weight_decay * self.current_factor)

    def on_fit_start(self, trainer, history):
        self._optimizer = trainer.optimizer
        self.original_weight_decays = [
            float(group.get("weight_decay", 0.0))
            for group in self._optimizer.param_groups
        ]
        next_epoch = int(history["completed_epochs"]) + 1
        self.current_factor = self._value_for_epoch(next_epoch)
        self._apply_weight_decay()

    def state_dict(self):
        return {
            "current_factor": self.current_factor,
            "original_weight_decays": list(self.original_weight_decays),
        }

    def load_state_dict(self, state):
        self.current_factor = float(state.get("current_factor", self.current_factor))
        state_weight_decays = state.get("original_weight_decays")
        if state_weight_decays is not None:
            self.original_weight_decays = [float(value) for value in state_weight_decays]
        if self._optimizer is not None and len(self.original_weight_decays) != len(self._optimizer.param_groups):
            raise ValueError("WeightDecayScheduler checkpoint state does not match optimizer param groups")
        self._apply_weight_decay()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        self.current_factor = self._value_for_epoch(epoch + 1)
        self._apply_weight_decay()

    def on_fit_end(self, trainer, history):
        del trainer, history
        if self._optimizer is None:
            return
        for group, original_weight_decay in zip(self._optimizer.param_groups, self.original_weight_decays):
            group["weight_decay"] = float(original_weight_decay)


class GradualUnfreezing(Callback):
    def __init__(self, module_name_groups, start_epoch=2, frequency=1):
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if frequency < 1:
            raise ValueError("frequency must be >= 1")
        self.module_name_groups = self._normalize_module_name_groups(module_name_groups)
        self.start_epoch = int(start_epoch)
        self.frequency = int(frequency)
        self.group_param_names = []
        self.original_requires_grad = {}
        self.unfrozen_group_count = 0
        self._param_lookup = {}

    def _normalize_module_name_groups(self, module_name_groups):
        if isinstance(module_name_groups, str):
            module_name_groups = [module_name_groups]
        groups = []
        for group in module_name_groups:
            if isinstance(group, str):
                group = (group,)
            else:
                group = tuple(group)
            if not group:
                raise ValueError("module_name_groups must not be empty")
            for name in group:
                if not isinstance(name, str) or not name:
                    raise ValueError("module_name_groups entries must be non-empty strings")
            groups.append(group)
        if not groups:
            raise ValueError("module_name_groups must not be empty")
        return tuple(groups)

    def _resolve_group_param_names(self, model):
        param_lookup = dict(model.named_parameters())
        group_param_names = []
        ownership = {}
        for group_index, group in enumerate(self.module_name_groups):
            resolved_names = []
            for prefix in group:
                matches = [
                    name
                    for name in param_lookup
                    if name == prefix or name.startswith(f"{prefix}.")
                ]
                if not matches:
                    raise ValueError(f"module prefix '{prefix}' matched no parameters")
                for name in matches:
                    if name in ownership and ownership[name] != group_index:
                        raise ValueError(f"module_name_groups overlap on parameter '{name}'")
                    ownership[name] = group_index
                    if name not in resolved_names:
                        resolved_names.append(name)
            group_param_names.append(tuple(resolved_names))
        return tuple(group_param_names), param_lookup

    def _target_unfrozen_group_count(self, next_epoch):
        if next_epoch < self.start_epoch:
            return 0
        return min(
            len(self.group_param_names),
            1 + (int(next_epoch) - self.start_epoch) // self.frequency,
        )

    def _apply_requires_grad_state(self):
        if not self._param_lookup:
            return
        for name in self.original_requires_grad:
            self._param_lookup[name].requires_grad = False
        for group_index in range(self.unfrozen_group_count):
            for name in self.group_param_names[group_index]:
                self._param_lookup[name].requires_grad = self.original_requires_grad[name]

    def on_fit_start(self, trainer, history):
        del history
        self.group_param_names, self._param_lookup = self._resolve_group_param_names(trainer.model)
        tracked_names = []
        for group in self.group_param_names:
            for name in group:
                if name not in tracked_names:
                    tracked_names.append(name)
        self.original_requires_grad = {
            name: self._param_lookup[name].requires_grad
            for name in tracked_names
        }
        self.unfrozen_group_count = self._target_unfrozen_group_count(next_epoch=1)
        self._apply_requires_grad_state()

    def state_dict(self):
        return {
            "original_requires_grad": dict(self.original_requires_grad),
            "unfrozen_group_count": self.unfrozen_group_count,
        }

    def load_state_dict(self, state):
        state_requires_grad = {
            name: bool(flag)
            for name, flag in state.get("original_requires_grad", {}).items()
        }
        if state_requires_grad:
            current_names = set(self.original_requires_grad)
            if current_names and set(state_requires_grad) != current_names:
                raise ValueError("GradualUnfreezing tracked parameters do not match checkpoint state")
            self.original_requires_grad = state_requires_grad
        self.unfrozen_group_count = int(state.get("unfrozen_group_count", 0))
        if self.unfrozen_group_count < 0 or self.unfrozen_group_count > len(self.group_param_names):
            raise ValueError("GradualUnfreezing checkpoint state is out of range")
        self._apply_requires_grad_state()

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary
        if epoch >= history["epochs"]:
            return
        target_count = self._target_unfrozen_group_count(next_epoch=epoch + 1)
        if target_count <= self.unfrozen_group_count:
            return
        self.unfrozen_group_count = target_count
        self._apply_requires_grad_state()


class ExponentialMovingAverage(Callback):
    def __init__(self, decay=0.999, apply_on_fit_end=False):
        if decay < 0.0 or decay >= 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = float(decay)
        self.apply_on_fit_end = apply_on_fit_end
        self.shadow_state = None
        self.num_updates = 0

    def on_fit_start(self, trainer, history):
        del history
        self.shadow_state = {
            name: tensor.detach().clone()
            for name, tensor in trainer.model.state_dict().items()
            if torch.is_floating_point(tensor)
        }
        self.num_updates = 0

    def on_after_optimizer_step(self, trainer, step):
        del step
        if self.shadow_state is None:
            return
        state_dict = trainer.model.state_dict()
        for name, shadow in self.shadow_state.items():
            current = state_dict[name].detach().to(device=shadow.device, dtype=shadow.dtype)
            shadow.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
        self.num_updates += 1

    def apply_to(self, model):
        if self.shadow_state is None:
            raise ValueError("ExponentialMovingAverage has no tracked state")
        state_dict = model.state_dict()
        for name, shadow in self.shadow_state.items():
            state_dict[name].copy_(shadow.to(device=state_dict[name].device, dtype=state_dict[name].dtype))

    def state_dict(self):
        if self.shadow_state is None:
            shadow_state = None
        else:
            shadow_state = {
                name: tensor.detach().clone()
                for name, tensor in self.shadow_state.items()
            }
        return {
            "shadow_state": shadow_state,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state):
        shadow_state = state.get("shadow_state")
        if shadow_state is None:
            self.shadow_state = None
        else:
            self.shadow_state = {
                name: tensor.detach().clone()
                for name, tensor in shadow_state.items()
            }
        self.num_updates = int(state.get("num_updates", 0))

    def on_fit_end(self, trainer, history):
        del history
        if self.apply_on_fit_end and self.shadow_state is not None:
            self.apply_to(trainer.model)


class Lookahead(Callback):
    def __init__(self, sync_period=5, slow_step_size=0.5):
        if sync_period < 1:
            raise ValueError("sync_period must be >= 1")
        if slow_step_size <= 0.0 or slow_step_size > 1.0:
            raise ValueError("slow_step_size must be in (0, 1]")
        self.sync_period = int(sync_period)
        self.slow_step_size = float(slow_step_size)
        self.slow_state = None
        self.step_count = 0

    def on_fit_start(self, trainer, history):
        del history
        self.slow_state = {
            name: tensor.detach().clone()
            for name, tensor in trainer.model.state_dict().items()
            if torch.is_floating_point(tensor)
        }
        self.step_count = 0

    def apply_to(self, model):
        if self.slow_state is None:
            raise ValueError("Lookahead has no tracked slow state")
        state_dict = model.state_dict()
        for name, slow in self.slow_state.items():
            state_dict[name].copy_(slow.to(device=state_dict[name].device, dtype=state_dict[name].dtype))

    def state_dict(self):
        if self.slow_state is None:
            slow_state = None
        else:
            slow_state = {
                name: tensor.detach().clone()
                for name, tensor in self.slow_state.items()
            }
        return {
            "slow_state": slow_state,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state):
        slow_state = state.get("slow_state")
        if slow_state is None:
            self.slow_state = None
        else:
            self.slow_state = {
                name: tensor.detach().clone()
                for name, tensor in slow_state.items()
            }
        self.step_count = int(state.get("step_count", 0))

    def on_after_optimizer_step(self, trainer, step):
        if self.slow_state is None:
            return
        self.step_count = int(step)
        if self.step_count % self.sync_period != 0:
            return
        state_dict = trainer.model.state_dict()
        for name, slow in self.slow_state.items():
            current = state_dict[name].detach().to(device=slow.device, dtype=slow.dtype)
            slow.add_(current - slow, alpha=self.slow_step_size)
        self.apply_to(trainer.model)


class StochasticWeightAveraging(Callback):
    def __init__(self, start_epoch=1, frequency=1, apply_on_fit_end=False):
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1")
        if frequency < 1:
            raise ValueError("frequency must be >= 1")
        self.start_epoch = int(start_epoch)
        self.frequency = int(frequency)
        self.apply_on_fit_end = apply_on_fit_end
        self.avg_state = None
        self.num_averaged = 0

    def on_fit_start(self, trainer, history):
        del trainer, history
        self.avg_state = None
        self.num_averaged = 0

    def _should_average(self, epoch):
        return epoch >= self.start_epoch and (epoch - self.start_epoch) % self.frequency == 0

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del train_summary, val_summary, history
        if not self._should_average(epoch):
            return
        current_state = {
            name: tensor.detach().clone()
            for name, tensor in trainer.model.state_dict().items()
            if torch.is_floating_point(tensor)
        }
        if self.avg_state is None:
            self.avg_state = current_state
            self.num_averaged = 1
            return

        next_count = self.num_averaged + 1
        for name, avg_tensor in self.avg_state.items():
            current_tensor = current_state[name].to(device=avg_tensor.device, dtype=avg_tensor.dtype)
            avg_tensor.add_(current_tensor - avg_tensor, alpha=1.0 / next_count)
        self.num_averaged = next_count

    def apply_to(self, model):
        if self.avg_state is None:
            raise ValueError("StochasticWeightAveraging has no averaged state")
        state_dict = model.state_dict()
        for name, avg_tensor in self.avg_state.items():
            state_dict[name].copy_(avg_tensor.to(device=state_dict[name].device, dtype=state_dict[name].dtype))

    def state_dict(self):
        if self.avg_state is None:
            avg_state = None
        else:
            avg_state = {
                name: tensor.detach().clone()
                for name, tensor in self.avg_state.items()
            }
        return {
            "avg_state": avg_state,
            "num_averaged": self.num_averaged,
        }

    def load_state_dict(self, state):
        avg_state = state.get("avg_state")
        if avg_state is None:
            self.avg_state = None
        else:
            self.avg_state = {
                name: tensor.detach().clone()
                for name, tensor in avg_state.items()
            }
        self.num_averaged = int(state.get("num_averaged", 0))

    def on_fit_end(self, trainer, history):
        del history
        if self.apply_on_fit_end and self.avg_state is not None:
            self.apply_to(trainer.model)


__all__ = [
    "Callback",
    "StopTraining",
    "EarlyStopping",
    "HistoryLogger",
    "AdaptiveGradientClipping",
    "GradientValueClipping",
    "GradientNoiseInjection",
    "GradientCentralization",
    "DeferredReweighting",
    "LabelSmoothingScheduler",
    "FocalGammaScheduler",
    "LogitAdjustTauScheduler",
    "LdamMarginScheduler",
    "PosWeightScheduler",
    "BootstrapBetaScheduler",
    "ConfidencePenaltyScheduler",
    "FloodingLevelScheduler",
    "GeneralizedCrossEntropyScheduler",
    "Poly1EpsilonScheduler",
    "SymmetricCrossEntropyBetaScheduler",
    "GradientAccumulationScheduler",
    "ModelCheckpoint",
    "WeightDecayScheduler",
    "GradualUnfreezing",
    "ExponentialMovingAverage",
    "Lookahead",
    "StochasticWeightAveraging",
]
