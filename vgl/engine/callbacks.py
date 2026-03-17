import torch

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
        del trainer
        record = {
            "epoch": epoch,
            "train": dict(train_summary),
            "val": None if val_summary is None else dict(val_summary),
            "best_epoch": history["best_epoch"],
            "best_metric": history["best_metric"],
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
    "GradientCentralization",
    "DeferredReweighting",
    "GradualUnfreezing",
    "ExponentialMovingAverage",
    "Lookahead",
    "StochasticWeightAveraging",
]
