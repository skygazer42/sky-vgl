from collections.abc import Iterable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

from vgl.engine.callbacks import Callback, StopTraining
from vgl.engine.checkpoints import CHECKPOINT_FORMAT, CHECKPOINT_FORMAT_VERSION
from vgl.engine.checkpoints import load_checkpoint as load_trainer_checkpoint
from vgl.engine.checkpoints import restore_checkpoint as restore_trainer_checkpoint
from vgl.engine.checkpoints import save_checkpoint as save_trainer_checkpoint
from vgl.engine.history import TrainingHistory
from vgl.engine.monitoring import extract_monitor_value, is_improvement, resolve_monitor
from vgl.metrics import build_metric
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    CHECKPOINT_FORMAT = CHECKPOINT_FORMAT
    CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    _SUPPORTED_PRECISIONS = {"32", "bf16-mixed", "fp16-mixed"}
    _SUPPORTED_SCHEDULER_INTERVALS = {"epoch", "step"}

    def __init__(
        self,
        model,
        task,
        optimizer,
        lr,
        max_epochs,
        metrics=None,
        monitor=None,
        monitor_mode=None,
        save_best_path=None,
        callbacks=None,
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        lr_scheduler=None,
        scheduler_monitor=None,
        optimizer_param_groups=None,
        lr_scheduler_interval="epoch",
        precision="32",
        grad_scaler=None,
    ):
        if accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be >= 1")
        if gradient_clip_val is not None and gradient_clip_val < 0:
            raise ValueError("gradient_clip_val must be >= 0")
        if scheduler_monitor is not None and lr_scheduler is None:
            raise ValueError("scheduler_monitor requires lr_scheduler")
        if lr_scheduler_interval not in self._SUPPORTED_SCHEDULER_INTERVALS:
            raise ValueError(
                f"lr_scheduler_interval must be one of {sorted(self._SUPPORTED_SCHEDULER_INTERVALS)}"
            )
        if precision not in self._SUPPORTED_PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(self._SUPPORTED_PRECISIONS)}")

        self.model = model
        self.task = task
        self.optimizer = optimizer(
            self._build_optimizer_param_groups(optimizer_param_groups, lr),
            lr=lr,
        )
        self.max_epochs = max_epochs
        metric_specs = getattr(task, "metrics", None) if metrics is None else metrics
        self.metric_specs = list(metric_specs or [])
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.save_best_path = Path(save_best_path) if save_best_path is not None else None
        self.callbacks = list(callbacks or [])
        self.best_state_dict = None
        self.best_epoch = None
        self.best_metric = None
        self.active_monitor = None
        self.global_step = 0
        self.accumulate_grad_batches = int(accumulate_grad_batches)
        self.gradient_clip_val = None if gradient_clip_val is None else float(gradient_clip_val)
        self.lr_scheduler = self._build_lr_scheduler(lr_scheduler)
        self.scheduler_monitor = scheduler_monitor
        self.lr_scheduler_interval = lr_scheduler_interval
        self.precision = precision
        self.grad_scaler = self._build_grad_scaler(grad_scaler)
        self._validate_scheduler_configuration()
        self._validate_optimizer_configuration()
        self._resume_state = None
        self.last_history = None

    @classmethod
    def save_checkpoint(cls, path, model_state_dict, *, metadata=None):
        return save_trainer_checkpoint(path, model_state_dict, metadata=metadata)

    @classmethod
    def load_checkpoint(cls, path, *, map_location=None, weights_only=True):
        return load_trainer_checkpoint(
            path,
            map_location=map_location,
            weights_only=weights_only,
        )

    @classmethod
    def restore_checkpoint(
        cls,
        model,
        path,
        *,
        map_location=None,
        strict=True,
        weights_only=True,
    ):
        return restore_trainer_checkpoint(
            model,
            path,
            map_location=map_location,
            strict=strict,
            weights_only=weights_only,
        )

    def _batches(self, data):
        if (
            hasattr(data, "nodes")
            or hasattr(data, "graphs")
            or hasattr(data, "labels")
            or hasattr(data, "x")
        ):
            return [data]
        if isinstance(data, Iterable):
            return data
        return [data]

    def _metrics(self):
        return [build_metric(metric) for metric in self.metric_specs]

    def _model_device_type(self):
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device.type
        buffer = next(self.model.buffers(), None)
        if buffer is not None:
            return buffer.device.type
        return "cpu"

    def _build_grad_scaler(self, grad_scaler):
        if grad_scaler is None:
            if self.precision == "fp16-mixed" and self._model_device_type() == "cuda":
                return torch.cuda.amp.GradScaler()
            return None
        if callable(grad_scaler) and not hasattr(grad_scaler, "scale"):
            grad_scaler = grad_scaler()
        for method_name in ("scale", "step", "update"):
            if not hasattr(grad_scaler, method_name):
                raise TypeError(f"grad_scaler must define {method_name}()")
        return grad_scaler

    def _build_lr_scheduler(self, lr_scheduler):
        if lr_scheduler is None:
            return None
        if callable(lr_scheduler) and not hasattr(lr_scheduler, "step"):
            lr_scheduler = lr_scheduler(self.optimizer)
        if not hasattr(lr_scheduler, "step"):
            raise TypeError("lr_scheduler must be a scheduler instance or a callable returning one")
        return lr_scheduler

    def _validate_scheduler_configuration(self):
        if self.lr_scheduler is None:
            return
        if self.lr_scheduler_interval != "step":
            return
        if self.scheduler_monitor is not None:
            raise ValueError("scheduler_monitor is not supported when lr_scheduler_interval='step'")
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            raise ValueError("ReduceLROnPlateau is not supported when lr_scheduler_interval='step'")

    def _validate_optimizer_configuration(self):
        if self._uses_sharpness_aware_optimizer() and self.grad_scaler is not None:
            raise ValueError("sharpness-aware optimizers are not supported with grad_scaler")

    def _build_optimizer_param_groups(self, optimizer_param_groups, lr):
        if optimizer_param_groups is None:
            return self.model.parameters()
        if callable(optimizer_param_groups):
            optimizer_param_groups = optimizer_param_groups(self.model, lr)
        if isinstance(optimizer_param_groups, dict):
            raise TypeError("optimizer_param_groups must be an iterable, not a single dict")
        if not isinstance(optimizer_param_groups, Iterable):
            raise TypeError(
                "optimizer_param_groups must be an iterable or a callable returning an iterable"
            )
        return optimizer_param_groups

    def _requires_unscaled_gradients(self):
        if self.gradient_clip_val is not None:
            return True
        for callback in self.callbacks:
            if type(callback).on_before_optimizer_step is not Callback.on_before_optimizer_step:
                return True
        return False

    def _uses_sharpness_aware_optimizer(self):
        return bool(
            getattr(self.optimizer, "supports_sharpness_aware_steps", False)
            or (
                hasattr(self.optimizer, "first_step")
                and callable(getattr(self.optimizer, "first_step"))
                and hasattr(self.optimizer, "second_step")
                and callable(getattr(self.optimizer, "second_step"))
            )
        )

    def _autocast_context(self):
        if self.precision == "32":
            return nullcontext()
        device_type = self._model_device_type()
        if self.precision == "bf16-mixed":
            dtype = torch.bfloat16
        else:
            if device_type != "cuda":
                raise ValueError("precision='fp16-mixed' requires a CUDA model/device")
            dtype = torch.float16
        return torch.autocast(device_type, dtype=dtype, enabled=True)

    def _run_callbacks(self, hook_name, **kwargs):
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if hook is not None:
                hook(self, **kwargs)

    def _forward_loss(self, batch, stage):
        with self._autocast_context():
            predictions = self.model(batch)
            loss = self.task.loss(batch, predictions, stage=stage)
        return predictions, loss

    def _uses_paired_forward_task(self):
        paired_loss = getattr(self.task, "paired_loss", None)
        return callable(paired_loss)

    def _forward_training_loss(self, batch, stage):
        if not self._uses_paired_forward_task():
            return self._forward_loss(batch, stage=stage)
        with self._autocast_context():
            predictions = self.model(batch)
            second_predictions = self.model(batch)
            loss = self.task.paired_loss(batch, predictions, second_predictions, stage=stage)
        return predictions, loss

    def _metric_inputs(self, batch, predictions, stage):
        metric_predictions = self.task.predictions_for_metrics(batch, predictions, stage=stage)
        targets = self.task.targets(batch, stage=stage)
        if metric_predictions.size(0) != targets.size(0):
            raise ValueError("Task metric predictions and targets must align in batch size")
        return metric_predictions, targets

    def _record_batch_summary(self, metrics, batch, predictions, loss, stage):
        metric_predictions, targets = self._metric_inputs(batch, predictions, stage=stage)
        count = int(targets.size(0))
        for metric in metrics:
            metric.update(metric_predictions.detach(), targets.detach())
        return loss.detach().item() * count, count

    def _backward_loss(self, loss):
        if self.grad_scaler is None:
            loss.backward()
            return
        self.grad_scaler.scale(loss).backward()

    def _unscale_gradients_if_needed(self):
        if self.grad_scaler is None or not self._requires_unscaled_gradients():
            return
        if not hasattr(self.grad_scaler, "unscale_"):
            raise TypeError(
                "grad_scaler must define unscale_() when pre-step gradient operations are enabled"
            )
        self.grad_scaler.unscale_(self.optimizer)

    def _run_before_optimizer_step(self, next_step):
        self._unscale_gradients_if_needed()
        self._run_callbacks("on_before_optimizer_step", step=next_step)
        if self.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

    def _run_after_second_backward(self):
        hook = getattr(self.optimizer, "on_after_second_backward", None)
        if hook is not None:
            hook()

    def _finish_optimizer_step(self, next_step):
        self.global_step = next_step
        self._step_lr_scheduler_on_step()
        self._run_callbacks("on_after_optimizer_step", step=self.global_step)
        self.optimizer.zero_grad()

    def _run_standard_training_group(self, group_batches, stage, metrics):
        group_size = len(group_batches)
        total_loss = 0.0
        total_items = 0
        for batch in group_batches:
            predictions, loss = self._forward_training_loss(batch, stage=stage)
            batch_loss, batch_items = self._record_batch_summary(metrics, batch, predictions, loss, stage=stage)
            total_loss += batch_loss
            total_items += batch_items
            self._backward_loss(loss / group_size)
        next_step = self.global_step + 1
        self._run_before_optimizer_step(next_step)
        if self.grad_scaler is None:
            self.optimizer.step()
        else:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        self._finish_optimizer_step(next_step)
        return total_loss, total_items

    def _run_sharpness_aware_training_group(self, group_batches, stage, metrics):
        group_size = len(group_batches)
        total_loss = 0.0
        total_items = 0
        for batch in group_batches:
            predictions, loss = self._forward_training_loss(batch, stage=stage)
            batch_loss, batch_items = self._record_batch_summary(metrics, batch, predictions, loss, stage=stage)
            total_loss += batch_loss
            total_items += batch_items
            self._backward_loss(loss / group_size)
        self.optimizer.first_step(zero_grad=True)
        for batch in group_batches:
            _, loss = self._forward_training_loss(batch, stage=stage)
            self._backward_loss(loss / group_size)
        self._run_after_second_backward()
        next_step = self.global_step + 1
        self._run_before_optimizer_step(next_step)
        self.optimizer.second_step(zero_grad=False)
        self._finish_optimizer_step(next_step)
        return total_loss, total_items

    def _callback_states(self):
        callback_counts = {}
        callback_states = []
        for callback in self.callbacks:
            state = callback.state_dict()
            if not state:
                continue
            callback_name = f"{callback.__class__.__module__}.{callback.__class__.__qualname__}"
            callback_index = callback_counts.get(callback_name, 0)
            callback_counts[callback_name] = callback_index + 1
            callback_states.append(
                {
                    "callback": callback_name,
                    "index": callback_index,
                    "state": state,
                }
            )
        return callback_states

    def _restore_callback_states(self, callback_states):
        if callback_states is None:
            return
        state_lookup = {
            (entry["callback"], int(entry.get("index", 0))): entry.get("state", {})
            for entry in callback_states
        }
        callback_counts = {}
        restored = set()
        for callback in self.callbacks:
            callback_name = f"{callback.__class__.__module__}.{callback.__class__.__qualname__}"
            callback_index = callback_counts.get(callback_name, 0)
            callback_counts[callback_name] = callback_index + 1
            key = (callback_name, callback_index)
            if key not in state_lookup:
                continue
            callback.load_state_dict(state_lookup[key])
            restored.add(key)
        missing = sorted(set(state_lookup) - restored)
        if missing:
            raise ValueError("checkpoint callback state does not match configured callbacks")

    def _grad_scaler_state_dict(self):
        if self.grad_scaler is None or not hasattr(self.grad_scaler, "state_dict"):
            return None
        return self.grad_scaler.state_dict()

    def _load_grad_scaler_state(self, state):
        if state is None:
            return
        if self.grad_scaler is None:
            raise ValueError("checkpoint contains grad_scaler_state_dict but trainer has no grad_scaler")
        if not hasattr(self.grad_scaler, "load_state_dict"):
            raise TypeError("grad_scaler must define load_state_dict() to restore checkpoint state")
        self.grad_scaler.load_state_dict(state)

    def save_training_checkpoint(self, path, *, history=None, metadata=None):
        history = history or self.last_history
        history_state = None if history is None else history.state_dict()
        return save_trainer_checkpoint(
            path,
            self.model.state_dict(),
            metadata=metadata,
            optimizer_state_dict=self.optimizer.state_dict(),
            lr_scheduler_state_dict=None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            grad_scaler_state_dict=self._grad_scaler_state_dict(),
            callback_states=self._callback_states(),
            trainer_state={
                "best_state_dict": None if self.best_state_dict is None else deepcopy(self.best_state_dict),
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "active_monitor": self.active_monitor,
                "global_step": self.global_step,
            },
            history_state=history_state,
        )

    def restore_training_checkpoint(
        self,
        path,
        *,
        map_location=None,
        strict=True,
        weights_only=True,
    ):
        payload = self.load_checkpoint(
            path,
            map_location=map_location,
            weights_only=weights_only,
        )
        self.model.load_state_dict(payload["model_state_dict"], strict=strict)
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        scheduler_state = payload.get("lr_scheduler_state_dict")
        if scheduler_state is not None:
            if self.lr_scheduler is None:
                raise ValueError("checkpoint contains lr_scheduler_state_dict but trainer has no lr_scheduler")
            self.lr_scheduler.load_state_dict(scheduler_state)
        self._load_grad_scaler_state(payload.get("grad_scaler_state_dict"))
        trainer_state = dict(payload.get("trainer_state") or {})
        self.best_state_dict = trainer_state.get("best_state_dict")
        self.best_epoch = trainer_state.get("best_epoch")
        self.best_metric = trainer_state.get("best_metric")
        self.active_monitor = trainer_state.get("active_monitor")
        self.global_step = int(trainer_state.get("global_step", 0))
        self._resume_state = {
            "history_state": payload.get("history_state"),
            "callback_states": payload.get("callback_states"),
            "trainer_state": trainer_state,
        }
        return payload

    def _run_epoch(self, data, stage, training):
        batches = list(self._batches(data))
        if not batches:
            raise ValueError(f"Trainer.{stage} requires at least one batch")

        metrics = self._metrics()
        for metric in metrics:
            metric.reset()

        total_loss = 0.0
        total_items = 0
        self.model.train(training)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            if training:
                self.optimizer.zero_grad()
            if training:
                for group_start in range(0, len(batches), self.accumulate_grad_batches):
                    group_end = min(group_start + self.accumulate_grad_batches, len(batches))
                    group_batches = batches[group_start:group_end]
                    if self._uses_sharpness_aware_optimizer():
                        batch_loss, batch_items = self._run_sharpness_aware_training_group(
                            group_batches,
                            stage=stage,
                            metrics=metrics,
                        )
                    else:
                        batch_loss, batch_items = self._run_standard_training_group(
                            group_batches,
                            stage=stage,
                            metrics=metrics,
                        )
                    total_loss += batch_loss
                    total_items += batch_items
            else:
                for batch in batches:
                    predictions, loss = self._forward_loss(batch, stage=stage)
                    batch_loss, batch_items = self._record_batch_summary(
                        metrics,
                        batch,
                        predictions,
                        loss,
                        stage=stage,
                    )
                    total_loss += batch_loss
                    total_items += batch_items

        if total_items == 0:
            raise ValueError(f"Trainer.{stage} requires at least one supervised example")

        summary = {"loss": total_loss / total_items}
        for metric in metrics:
            value = metric.compute()
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric {metric.name} must return a scalar value")
            summary[metric.name] = float(value)
        return summary

    def _step_lr_scheduler(self, monitor, train_summary, val_summary):
        if self.lr_scheduler is None:
            return
        if self.lr_scheduler_interval != "epoch":
            return
        scheduler_monitor = self.scheduler_monitor
        if scheduler_monitor is None and isinstance(self.lr_scheduler, ReduceLROnPlateau):
            scheduler_monitor = monitor
        if scheduler_monitor is None:
            self.lr_scheduler.step()
            return
        value = extract_monitor_value(
            scheduler_monitor,
            train_summary=train_summary,
            val_summary=val_summary,
            error_subject="Scheduler monitor",
        )
        self.lr_scheduler.step(value)

    def _step_lr_scheduler_on_step(self):
        if self.lr_scheduler is None:
            return
        if self.lr_scheduler_interval != "step":
            return
        self.lr_scheduler.step()

    def _resolve_monitor(self, val_data):
        return resolve_monitor(
            self.monitor,
            has_val_data=val_data is not None,
            mode=self.monitor_mode,
        )

    def _monitor_value(self, monitor, train_summary, val_summary):
        return extract_monitor_value(
            monitor,
            train_summary=train_summary,
            val_summary=val_summary,
        )

    def _save_best(self):
        if self.save_best_path is None:
            return
        self.save_checkpoint(
            self.save_best_path,
            self.best_state_dict,
            metadata={
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "monitor": self.active_monitor,
            },
        )

    def evaluate(self, data, stage="val"):
        return self._run_epoch(data, stage=stage, training=False)

    def test(self, data):
        return self.evaluate(data, stage="test")

    def fit(self, train_data, val_data=None):
        monitor, mode = self._resolve_monitor(val_data)
        resume_state = self._resume_state
        if resume_state is None or resume_state.get("history_state") is None:
            history = TrainingHistory(epochs=self.max_epochs, monitor=monitor)
            self.active_monitor = monitor
            self.global_step = 0
            start_epoch = 1
        else:
            history = TrainingHistory.from_state_dict(resume_state["history_state"])
            history["epochs"] = self.max_epochs
            history["monitor"] = monitor
            history["stopped_early"] = False
            history["stop_reason"] = None
            start_epoch = int(history["completed_epochs"]) + 1
            trainer_state = resume_state.get("trainer_state", {})
            self.best_state_dict = trainer_state.get("best_state_dict")
            self.best_epoch = trainer_state.get("best_epoch")
            self.best_metric = trainer_state.get("best_metric")
            self.active_monitor = trainer_state.get("active_monitor", monitor) or monitor
            self.global_step = int(trainer_state.get("global_step", 0))
        self._run_callbacks("on_fit_start", history=history)
        if resume_state is not None:
            self._restore_callback_states(resume_state.get("callback_states"))
            self._resume_state = None

        for epoch in range(start_epoch, self.max_epochs + 1):
            train_summary = self._run_epoch(train_data, stage="train", training=True)
            val_summary = None
            if val_data is not None:
                val_summary = self._run_epoch(val_data, stage="val", training=False)

            current = self._monitor_value(monitor, train_summary, val_summary)
            improved = is_improvement(
                current,
                self.best_metric,
                mode,
            )
            if improved:
                self.best_metric = float(current)
                self.best_epoch = epoch
                self.best_state_dict = deepcopy(self.model.state_dict())
                self._save_best()

            history.record_epoch(
                epoch=epoch,
                train_summary=train_summary,
                val_summary=val_summary,
                best_epoch=self.best_epoch,
                best_metric=self.best_metric,
            )
            self._step_lr_scheduler(monitor, train_summary, val_summary)
            try:
                self._run_callbacks(
                    "on_epoch_end",
                    epoch=epoch,
                    train_summary=train_summary,
                    val_summary=val_summary,
                    history=history,
                )
            except StopTraining as exc:
                history.mark_stopped(str(exc) or None)
                break

        history.finalize(best_epoch=self.best_epoch, best_metric=self.best_metric)
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        self._run_callbacks("on_fit_end", history=history)
        self.last_history = history
        return history
