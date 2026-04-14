from collections.abc import Iterable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter

from vgl.engine.callbacks import Callback, StopTraining
from vgl.engine.checkpoints import CHECKPOINT_FORMAT, CHECKPOINT_FORMAT_VERSION
from vgl.engine.checkpoints import checkpoint_event_fields
from vgl.engine.checkpoints import load_checkpoint as load_trainer_checkpoint
from vgl.engine.checkpoints import restore_checkpoint as restore_trainer_checkpoint
from vgl.engine.checkpoints import save_checkpoint as save_trainer_checkpoint
from vgl.engine.history import (
    PROFILE_COUNT_KEYS,
    PROFILE_TOTAL_KEYS,
    TrainingHistory,
    normalize_profile,
)
from vgl.engine.logging import ConsoleLogger
from vgl.engine.monitoring import extract_monitor_value, is_improvement, resolve_monitor
from vgl.graph import Graph, GraphBatch, GraphView, LinkPredictionBatch, NodeBatch, TemporalEventBatch
from vgl.metrics import build_metric
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


_SUPPORTED_PRECISIONS = {"32", "bf16-mixed", "fp16-mixed"}
_SUPPORTED_SCHEDULER_INTERVALS = {"epoch", "step"}


def _validate_init_config(
    *,
    accumulate_grad_batches,
    log_every_n_steps,
    gradient_clip_val,
    console_flush_every_n_steps,
    scheduler_monitor,
    lr_scheduler,
    lr_scheduler_interval,
    precision,
    non_blocking,
    num_sanity_val_steps,
    profiler,
):
    if accumulate_grad_batches < 1:
        raise ValueError("accumulate_grad_batches must be >= 1")
    if log_every_n_steps < 1:
        raise ValueError("log_every_n_steps must be >= 1")
    if gradient_clip_val is not None and gradient_clip_val < 0:
        raise ValueError("gradient_clip_val must be >= 0")
    if console_flush_every_n_steps is not None and console_flush_every_n_steps < 1:
        raise ValueError("console_flush_every_n_steps must be >= 1")
    if scheduler_monitor is not None and lr_scheduler is None:
        raise ValueError("scheduler_monitor requires lr_scheduler")
    if lr_scheduler_interval not in _SUPPORTED_SCHEDULER_INTERVALS:
        raise ValueError(
            f"lr_scheduler_interval must be one of {sorted(_SUPPORTED_SCHEDULER_INTERVALS)}"
        )
    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(_SUPPORTED_PRECISIONS)}")
    if non_blocking is not None and not isinstance(non_blocking, bool):
        raise TypeError("non_blocking must be None or a bool")
    if num_sanity_val_steps < 0:
        raise ValueError("num_sanity_val_steps must be >= 0")
    if profiler not in {None, "simple"}:
        raise ValueError("profiler must be None or 'simple'")


class Trainer:
    CHECKPOINT_FORMAT = CHECKPOINT_FORMAT
    CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    _SUPPORTED_PRECISIONS = _SUPPORTED_PRECISIONS
    _SUPPORTED_SCHEDULER_INTERVALS = _SUPPORTED_SCHEDULER_INTERVALS
    _SUPPORTED_VGL_TRANSFER_TYPES = (
        Graph,
        GraphView,
        GraphBatch,
        NodeBatch,
        LinkPredictionBatch,
        TemporalEventBatch,
    )

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
        loggers=None,
        log_every_n_steps=50,
        enable_console_logging=True,
        enable_progress_bar=True,
        console_flush_every_n_steps=None,
        console_mode="detailed",
        console_metric_names=None,
        console_show_learning_rate=True,
        console_show_events=True,
        console_show_timestamp=True,
        console_theme="default",
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        lr_scheduler=None,
        scheduler_monitor=None,
        optimizer_param_groups=None,
        lr_scheduler_interval="epoch",
        precision="32",
        device=None,
        move_batch_to_device=True,
        non_blocking=None,
        grad_scaler=None,
        default_root_dir=None,
        run_name=None,
        fast_dev_run=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        profiler=None,
    ):
        _validate_init_config(
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            console_flush_every_n_steps=console_flush_every_n_steps,
            scheduler_monitor=scheduler_monitor,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=lr_scheduler_interval,
            precision=precision,
            non_blocking=non_blocking,
            num_sanity_val_steps=num_sanity_val_steps,
            profiler=profiler,
        )

        self.model = model
        self.device = None if device is None else torch.device(device)
        self.move_batch_to_device = bool(move_batch_to_device)
        self.non_blocking = non_blocking
        if self.device is not None:
            self.model.to(self.device)
        self.task = task
        self.default_root_dir = None if default_root_dir is None else Path(default_root_dir)
        self.run_name = None if run_name is None else str(run_name)
        self._fast_dev_run_batches = self._normalize_fast_dev_run(fast_dev_run)
        self.fast_dev_run = self._fast_dev_run_batches is not None
        self.limit_train_batches = self._normalize_batch_limit(limit_train_batches, name="limit_train_batches")
        self.limit_val_batches = self._normalize_batch_limit(limit_val_batches, name="limit_val_batches")
        self.limit_test_batches = self._normalize_batch_limit(limit_test_batches, name="limit_test_batches")
        self.val_check_interval = self._normalize_val_check_interval(val_check_interval)
        self.num_sanity_val_steps = int(num_sanity_val_steps)
        self.profiler = profiler
        self.optimizer = optimizer(
            self._build_optimizer_param_groups(optimizer_param_groups, lr),
            lr=lr,
        )
        self.max_epochs = max_epochs
        metric_specs = getattr(task, "metrics", None) if metrics is None else metrics
        self.metric_specs = list(metric_specs or [])
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.callbacks = list(callbacks or [])
        self._resolve_callback_artifact_locations(self.callbacks)
        self.log_every_n_steps = int(log_every_n_steps)
        self.loggers = self._build_loggers(
            loggers,
            enable_console_logging=enable_console_logging,
            enable_progress_bar=enable_progress_bar,
            console_flush_every_n_steps=console_flush_every_n_steps,
            console_mode=console_mode,
            console_metric_names=console_metric_names,
            console_show_learning_rate=console_show_learning_rate,
            console_show_events=console_show_events,
            console_show_timestamp=console_show_timestamp,
            console_theme=console_theme,
        )
        self._resolve_logger_artifact_locations(self.loggers)
        self.save_best_path = self._resolve_artifact_path(save_best_path)
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
        self._fit_start_time = None
        self._active_epoch = None
        self._active_stage = None
        self._active_batch_idx = None
        self._epoch_total_batches = None
        self._epoch_start_time = None
        self._fit_epochs = int(max_epochs)
        self._checkpointing_enabled = True
        self._fit_profile = self._empty_profile()
        self._epoch_profile = None

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

    def _resolved_device_type(self):
        if self.device is not None:
            return self.device.type
        return self._model_device_type()

    def _resolved_non_blocking(self, batch):
        if self.non_blocking is not None:
            return self.non_blocking
        if not isinstance(batch, torch.Tensor):
            return False
        return self._resolved_device_type() == "cuda" and batch.device.type == "cpu"

    def _move_batch_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=self._resolved_non_blocking(batch))
        if isinstance(batch, dict):
            return {key: self._move_batch_to_device(value) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self._move_batch_to_device(item) for item in batch]
        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(item) for item in batch)
        if isinstance(batch, self._SUPPORTED_VGL_TRANSFER_TYPES):
            return self._move_vgl_batch_to_device(batch)
        if batch is None or isinstance(batch, (str, bytes, int, float, bool)):
            return batch
        raise TypeError(f"Unsupported batch type for automatic device transfer: {type(batch)!r}")

    def _move_vgl_batch_to_device(self, batch):
        if hasattr(batch, "to") and callable(batch.to):
            non_blocking = self._resolved_non_blocking(batch)
            try:
                return batch.to(self.device, non_blocking=non_blocking)
            except TypeError:
                return batch.to(self.device)
        if is_dataclass(batch):
            values = {
                field.name: self._move_vgl_value_to_device(getattr(batch, field.name))
                for field in fields(batch)
            }
            return type(batch)(**values)
        raise TypeError(f"VGL batch type does not support device transfer: {type(batch)!r}")

    def _move_vgl_value_to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=self._resolved_non_blocking(value))
        if isinstance(value, dict):
            return {key: self._move_vgl_value_to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_vgl_value_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_vgl_value_to_device(item) for item in value)
        if isinstance(value, self._SUPPORTED_VGL_TRANSFER_TYPES):
            return self._move_vgl_batch_to_device(value)
        if is_dataclass(value):
            values = {
                field.name: self._move_vgl_value_to_device(getattr(value, field.name))
                for field in fields(value)
            }
            return type(value)(**values)
        return value

    def _prepare_batch(self, batch):
        if self.device is None:
            return batch
        if not self.move_batch_to_device:
            return batch
        return self._move_batch_to_device(batch)

    def _build_grad_scaler(self, grad_scaler):
        if grad_scaler is None:
            if self.precision == "fp16-mixed" and self._resolved_device_type() == "cuda":
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

    def _normalize_fast_dev_run(self, fast_dev_run):
        if isinstance(fast_dev_run, bool):
            return 1 if fast_dev_run else None
        if isinstance(fast_dev_run, int):
            if fast_dev_run < 1:
                raise ValueError("fast_dev_run must be >= 1")
            return int(fast_dev_run)
        raise TypeError("fast_dev_run must be a bool or an integer")

    def _normalize_batch_limit(self, value, *, name):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an int or float")
        if isinstance(value, int):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
            return int(value)
        if isinstance(value, float):
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in (0.0, 1.0]")
            return float(value)
        raise TypeError(f"{name} must be an int or float")

    def _normalize_val_check_interval(self, value):
        if isinstance(value, bool):
            raise TypeError("val_check_interval must be an int or float")
        if isinstance(value, int):
            if value < 1:
                raise ValueError("val_check_interval must be >= 1")
            return int(value)
        if isinstance(value, float):
            if value <= 0.0 or value > 1.0:
                raise ValueError("val_check_interval must be in (0.0, 1.0]")
            return float(value)
        raise TypeError("val_check_interval must be an int or float")

    def _resolve_artifact_path(self, path):
        if path is None:
            return None
        artifact_path = Path(path)
        if self.default_root_dir is not None and not artifact_path.is_absolute():
            artifact_path = self.default_root_dir / artifact_path
        return artifact_path

    def _resolve_logger_artifact_locations(self, loggers):
        for logger in loggers:
            for attribute in ("path", "log_dir"):
                value = getattr(logger, attribute, None)
                if value is None:
                    continue
                resolved = self._resolve_artifact_path(value)
                if resolved is not None:
                    setattr(logger, attribute, resolved)

    def _resolve_callback_artifact_locations(self, callbacks):
        for callback in callbacks:
            dirpath = getattr(callback, "dirpath", None)
            if dirpath is None:
                continue
            resolved = self._resolve_artifact_path(dirpath)
            if resolved is not None:
                callback.dirpath = resolved

    def _effective_epochs(self):
        if self.fast_dev_run:
            return 1
        return int(self.max_epochs)

    def _limit_setting_for_stage(self, stage):
        if stage in {"val", "sanity_val"}:
            return self.limit_val_batches
        if stage == "test":
            return self.limit_test_batches
        return self.limit_train_batches

    def _limited_batch_count(self, total_batches, *, stage, override=None):
        if total_batches < 1:
            return 0
        limit = self._limit_setting_for_stage(stage)
        if isinstance(limit, float):
            count = max(1, int(total_batches * limit))
        else:
            count = min(total_batches, int(limit))
        if override is not None:
            count = min(count, int(override))
        if self._fast_dev_run_batches is not None:
            count = min(count, int(self._fast_dev_run_batches))
        return max(1, count)

    def _apply_batch_limit(self, batches, *, stage, override=None):
        if not batches:
            return batches
        count = self._limited_batch_count(len(batches), stage=stage, override=override)
        return list(batches[:count])

    def _mid_epoch_validation_points(self, total_batches):
        if total_batches <= 1:
            return set()
        interval = self.val_check_interval
        if isinstance(interval, int):
            if interval >= total_batches:
                return set()
            return set(range(interval, total_batches, interval))
        if interval >= 1.0:
            return set()
        step = max(1, int(total_batches * interval))
        return set(range(step, total_batches, step))

    def _empty_profile(self):
        profile = {key: 0.0 for key in PROFILE_TOTAL_KEYS}
        profile.update({key: 0 for key in PROFILE_COUNT_KEYS})
        return profile

    def _profile_add(self, key, value):
        if self.profiler != "simple":
            return
        self._fit_profile[key] += value
        if self._epoch_profile is not None and key in self._epoch_profile:
            self._epoch_profile[key] += value

    def _profile_snapshot(self, profile):
        return normalize_profile(profile, profiler=self.profiler)

    def _record_with_run_context(self, record):
        record["run_name"] = self.run_name
        record["root_dir"] = None if self.default_root_dir is None else str(self.default_root_dir)
        record["fast_dev_run"] = self.fast_dev_run
        record["profiler"] = self.profiler
        return record

    def _record_with_profile(self, record, profile):
        record = self._record_with_run_context(record)
        if self.profiler == "simple":
            record["profile"] = self._profile_snapshot(profile)
        return record

    def _artifact_paths(self):
        logger_paths = []
        for logger in self.loggers:
            for attribute in ("path", "log_dir"):
                value = getattr(logger, attribute, None)
                if value is None:
                    continue
                logger_paths.append(
                    {
                        "logger": logger.__class__.__name__,
                        "path": str(value),
                    }
                )
                break
        callback_dirs = []
        for callback in self.callbacks:
            dirpath = getattr(callback, "dirpath", None)
            if dirpath is None:
                continue
            callback_dirs.append(
                {
                    "callback": callback.__class__.__name__,
                    "dirpath": str(dirpath),
                }
            )
        return {
            "save_best_path": None if self.save_best_path is None else str(self.save_best_path),
            "logger_paths": logger_paths,
            "callback_dirs": callback_dirs,
        }

    def _record_with_artifact_paths(self, record):
        record["artifact_paths"] = self._artifact_paths()
        return record

    def _build_loggers(
        self,
        loggers,
        *,
        enable_console_logging,
        enable_progress_bar,
        console_flush_every_n_steps,
        console_mode,
        console_metric_names,
        console_show_learning_rate,
        console_show_events,
        console_show_timestamp,
        console_theme,
    ):
        if loggers is None:
            logger_list = []
        elif isinstance(loggers, Iterable) and not isinstance(loggers, (str, bytes, dict)):
            logger_list = list(loggers)
        else:
            logger_list = [loggers]
        required_methods = (
            "on_fit_start",
            "on_train_step",
            "on_epoch_end",
            "on_evaluate_end",
            "on_fit_end",
            "on_exception",
            "on_event",
            "finalize",
        )
        for logger in logger_list:
            for method_name in required_methods:
                if not hasattr(logger, method_name):
                    raise TypeError(f"logger must define {method_name}()")
        if enable_console_logging and not any(isinstance(logger, ConsoleLogger) for logger in logger_list):
            logger_list.insert(
                0,
                ConsoleLogger(
                    enable_progress_bar=enable_progress_bar,
                    flush_every_n_steps=console_flush_every_n_steps,
                    mode=console_mode,
                    metric_names=console_metric_names,
                    show_learning_rate=console_show_learning_rate,
                    show_events=console_show_events,
                    show_timestamp=console_show_timestamp,
                    theme=console_theme,
                ),
            )
        return logger_list

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
        device_type = self._resolved_device_type()
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

    def _run_loggers(self, hook_name, payload, *, suppress_errors=False):
        try:
            for logger in self.loggers:
                hook = getattr(logger, hook_name, None)
                if hook is not None:
                    hook(payload)
        except Exception:
            if not suppress_errors:
                raise

    def _finalize_loggers(self, status, *, suppress_errors=False):
        try:
            for logger in self.loggers:
                logger.finalize(status)
        except Exception:
            if not suppress_errors:
                raise

    def _fit_elapsed_seconds(self):
        if self._fit_start_time is None:
            return None
        return float(perf_counter() - self._fit_start_time)

    def _steps_per_second(self):
        elapsed_seconds = self._fit_elapsed_seconds()
        if elapsed_seconds is None or elapsed_seconds <= 0.0 or self.global_step <= 0:
            return None
        return float(self.global_step / elapsed_seconds)

    def _epoch_elapsed_seconds(self):
        if self._epoch_start_time is None:
            return None
        return float(perf_counter() - self._epoch_start_time)

    def _parameter_counts(self):
        total_parameters = 0
        trainable_parameters = 0
        for parameter in self.model.parameters():
            count = int(parameter.numel())
            total_parameters += count
            if parameter.requires_grad:
                trainable_parameters += count
        return total_parameters, trainable_parameters

    def _learning_rate_metrics(self):
        param_groups = getattr(self.optimizer, "param_groups", None)
        if not param_groups:
            return {}
        if len(param_groups) == 1:
            return {"lr": float(param_groups[0]["lr"])}
        return {
            f"lr/group_{index}": float(group["lr"])
            for index, group in enumerate(param_groups)
        }

    def _summary_metrics(self, summary, *, stage):
        return {f"{stage}_{key}": float(value) for key, value in summary.items()}

    def _summary_from_metrics(self, metrics, total_loss, total_items, *, stage):
        if total_items == 0:
            raise ValueError(f"Trainer.{stage} requires at least one supervised example")
        summary = {"loss": total_loss / total_items}
        for metric in metrics:
            value = metric.compute()
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric {metric.name} must return a scalar value")
            summary[metric.name] = float(value)
        return summary

    def _build_fit_start_record(self, monitor):
        total_parameters, trainable_parameters = self._parameter_counts()
        return self._record_with_artifact_paths(self._record_with_run_context(
            {
            "event": "fit_start",
            "stage": "fit",
            "epoch": self._active_epoch,
            "epochs": self._fit_epochs,
            "global_step": self.global_step,
            "batch_idx": None,
            "metrics": {},
            "monitor": monitor,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "elapsed_seconds": 0.0,
            "precision": self.precision,
            "model_name": self.model.__class__.__name__,
            "task_name": self.task.__class__.__name__,
            "optimizer_name": self.optimizer.__class__.__name__,
            "lr_scheduler_name": None if self.lr_scheduler is None else self.lr_scheduler.__class__.__name__,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "callback_names": [callback.__class__.__name__ for callback in self.callbacks],
            "logger_names": [logger.__class__.__name__ for logger in self.loggers],
            "num_sanity_val_steps": self.num_sanity_val_steps,
            "limit_train_batches": self.limit_train_batches,
            "limit_val_batches": self.limit_val_batches,
            "limit_test_batches": self.limit_test_batches,
            "val_check_interval": self.val_check_interval,
        }
        ))

    def _build_stage_start_record(self, stage, *, total_batches):
        return self._record_with_run_context(
            {
            "event": "stage_start",
            "stage": stage,
            "epoch": self._active_epoch,
            "epochs": self._fit_epochs,
            "global_step": self.global_step,
            "batch_idx": None,
            "metrics": {},
            "monitor": self.active_monitor,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "elapsed_seconds": self._fit_elapsed_seconds() or 0.0,
            "total_batches": int(total_batches),
        }
        )

    def _build_train_step_record(self, summary):
        metrics = {key: float(value) for key, value in summary.items()}
        metrics.update(self._learning_rate_metrics())
        return self._record_with_profile(
            {
            "event": "train_step",
            "stage": "train",
            "epoch": self._active_epoch,
            "epochs": self._fit_epochs,
            "global_step": self.global_step,
            "batch_idx": self._active_batch_idx,
            "metrics": metrics,
            "monitor": self.active_monitor,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "elapsed_seconds": self._fit_elapsed_seconds(),
            "epoch_elapsed_seconds": self._epoch_elapsed_seconds(),
            "epoch_total_batches": self._epoch_total_batches,
            "steps_per_second": self._steps_per_second(),
        },
            self._epoch_profile,
        )

    def _build_epoch_record(self, train_summary, val_summary, *, epoch_elapsed_seconds):
        metrics = self._summary_metrics(train_summary, stage="train")
        if val_summary is not None:
            metrics.update(self._summary_metrics(val_summary, stage="val"))
        metrics.update(self._learning_rate_metrics())
        return self._record_with_profile(
            {
            "event": "epoch_end",
            "stage": "fit",
            "epoch": self._active_epoch,
            "epochs": self._fit_epochs,
            "global_step": self.global_step,
            "batch_idx": None,
            "metrics": metrics,
            "monitor": self.active_monitor,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "elapsed_seconds": float(epoch_elapsed_seconds),
        },
            self._epoch_profile,
        )

    def _build_stage_record(self, summary, *, stage, elapsed_seconds):
        return self._record_with_profile(
            {
            "event": "evaluate_end",
            "stage": stage,
            "epoch": self._active_epoch,
            "epochs": self._fit_epochs,
            "global_step": self.global_step,
            "batch_idx": None,
            "metrics": self._summary_metrics(summary, stage=stage),
            "monitor": self.active_monitor,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "elapsed_seconds": float(elapsed_seconds),
        },
            self._epoch_profile,
        )

    def _build_fit_end_record(self, history):
        metrics = {}
        final_train = history.get("final_train")
        final_val = history.get("final_val")
        if final_train is not None:
            metrics.update(self._summary_metrics(final_train, stage="train"))
        if final_val is not None:
            metrics.update(self._summary_metrics(final_val, stage="val"))
        metrics.update(self._learning_rate_metrics())
        epoch_durations = [
            float(duration)
            for duration in history.get("epoch_elapsed_seconds", [])
            if duration is not None
        ]
        average_epoch_seconds = None
        if epoch_durations:
            average_epoch_seconds = float(sum(epoch_durations) / len(epoch_durations))
        average_steps_per_second = None
        fit_elapsed_seconds = history["fit_elapsed_seconds"] or 0.0
        if fit_elapsed_seconds > 0.0 and self.global_step > 0:
            average_steps_per_second = float(self.global_step / fit_elapsed_seconds)
        return self._record_with_artifact_paths(self._record_with_profile(
            {
            "event": "fit_end",
            "stage": "fit",
            "epoch": history["completed_epochs"],
            "epochs": self._fit_epochs,
            "global_step": self.global_step,
            "batch_idx": None,
            "metrics": metrics,
            "monitor": history["monitor"],
            "best_epoch": history["best_epoch"],
            "best_metric": history["best_metric"],
            "elapsed_seconds": fit_elapsed_seconds,
            "average_epoch_seconds": average_epoch_seconds,
            "average_steps_per_second": average_steps_per_second,
            "stopped_early": history["stopped_early"],
            "stop_reason": history["stop_reason"],
        },
            self._fit_profile,
        ))

    def _build_exception_record(self, exception):
        return self._record_with_run_context(
            {
            "event": "exception",
            "stage": self._active_stage,
            "epoch": self._active_epoch,
            "epochs": self._fit_epochs,
            "global_step": self.global_step,
            "batch_idx": self._active_batch_idx,
            "metrics": {},
            "monitor": self.active_monitor,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "elapsed_seconds": self._fit_elapsed_seconds() or 0.0,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
        }
        )

    def _build_custom_event_record(self, event, *, stage=None, metrics=None, **fields):
        record = self._record_with_run_context({
            "event": str(event),
            "stage": self._active_stage if stage is None else stage,
            "epoch": self._active_epoch,
            "epochs": self._fit_epochs,
            "global_step": self.global_step,
            "batch_idx": self._active_batch_idx,
            "metrics": {} if metrics is None else {key: float(value) for key, value in metrics.items()},
            "monitor": self.active_monitor,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "elapsed_seconds": self._fit_elapsed_seconds() or 0.0,
        })
        for key, value in fields.items():
            if isinstance(value, Path):
                value = str(value)
            record[key] = value
        if str(event) == "checkpoint_saved":
            record = self._record_with_artifact_paths(record)
        return record

    def log_event(self, event, *, stage=None, metrics=None, **fields):
        self._run_loggers(
            "on_event",
            self._build_custom_event_record(
                event,
                stage=stage,
                metrics=metrics,
                **fields,
            ),
        )

    def _forward_loss(self, batch, stage):
        forward_start_time = perf_counter()
        with self._autocast_context():
            predictions = self.model(batch)
            loss = self.task.loss(batch, predictions, stage=stage)
        self._profile_add("forward_seconds_total", perf_counter() - forward_start_time)
        return predictions, loss

    def _uses_paired_forward_task(self):
        paired_loss = getattr(self.task, "paired_loss", None)
        return callable(paired_loss)

    def _forward_training_loss(self, batch, stage):
        if not self._uses_paired_forward_task():
            return self._forward_loss(batch, stage=stage)
        forward_start_time = perf_counter()
        with self._autocast_context():
            predictions = self.model(batch)
            second_predictions = self.model(batch)
            loss = self.task.paired_loss(batch, predictions, second_predictions, stage=stage)
        self._profile_add("forward_seconds_total", perf_counter() - forward_start_time)
        return predictions, loss

    def _metric_inputs(self, batch, predictions, stage):
        metric_predictions = self.task.predictions_for_metrics(batch, predictions, stage=stage)
        targets = self.task.targets(batch, stage=stage)
        if metric_predictions.size(0) != targets.size(0):
            raise ValueError("Task metric predictions and targets must align in batch size")
        return metric_predictions, targets

    def _record_batch_summary(self, metrics, batch, predictions, loss, stage, extra_metrics=None):
        metric_predictions, targets = self._metric_inputs(batch, predictions, stage=stage)
        count = int(targets.size(0))
        for metric in metrics:
            metric.update(metric_predictions.detach(), targets.detach(), batch=batch)
        for metric in extra_metrics or []:
            metric.update(metric_predictions.detach(), targets.detach(), batch=batch)
        return float(loss.detach()) * count, count

    def _backward_loss(self, loss):
        backward_start_time = perf_counter()
        if self.grad_scaler is None:
            loss.backward()
        else:
            self.grad_scaler.scale(loss).backward()
        self._profile_add("backward_seconds_total", perf_counter() - backward_start_time)

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
        train_step_start_time = perf_counter()
        prepared_batches = [self._prepare_batch(batch) for batch in group_batches]
        group_size = len(prepared_batches)
        total_loss = 0.0
        total_items = 0
        step_metrics = self._metrics()
        for batch in prepared_batches:
            predictions, loss = self._forward_training_loss(batch, stage=stage)
            batch_loss, batch_items = self._record_batch_summary(
                metrics,
                batch,
                predictions,
                loss,
                stage=stage,
                extra_metrics=step_metrics,
            )
            total_loss += batch_loss
            total_items += batch_items
            self._backward_loss(loss / group_size)
        next_step = self.global_step + 1
        optimizer_step_start_time = perf_counter()
        self._run_before_optimizer_step(next_step)
        if self.grad_scaler is None:
            self.optimizer.step()
        else:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        self._finish_optimizer_step(next_step)
        self._profile_add(
            "optimizer_step_seconds_total",
            perf_counter() - optimizer_step_start_time,
        )
        self._profile_add("train_step_seconds_total", perf_counter() - train_step_start_time)
        self._profile_add("train_step_count", 1)
        step_summary = self._summary_from_metrics(step_metrics, total_loss, total_items, stage=stage)
        if self.global_step % self.log_every_n_steps == 0:
            self._run_loggers("on_train_step", self._build_train_step_record(step_summary))
        return total_loss, total_items

    def _run_sharpness_aware_training_group(self, group_batches, stage, metrics):
        train_step_start_time = perf_counter()
        prepared_batches = [self._prepare_batch(batch) for batch in group_batches]
        group_size = len(prepared_batches)
        total_loss = 0.0
        total_items = 0
        step_metrics = self._metrics()
        for batch in prepared_batches:
            predictions, loss = self._forward_training_loss(batch, stage=stage)
            batch_loss, batch_items = self._record_batch_summary(
                metrics,
                batch,
                predictions,
                loss,
                stage=stage,
                extra_metrics=step_metrics,
            )
            total_loss += batch_loss
            total_items += batch_items
            self._backward_loss(loss / group_size)
        optimizer_step_start_time = perf_counter()
        self.optimizer.first_step(zero_grad=True)
        for batch in prepared_batches:
            _, loss = self._forward_training_loss(batch, stage=stage)
            self._backward_loss(loss / group_size)
        self._run_after_second_backward()
        next_step = self.global_step + 1
        self._run_before_optimizer_step(next_step)
        self.optimizer.second_step(zero_grad=False)
        self._finish_optimizer_step(next_step)
        self._profile_add(
            "optimizer_step_seconds_total",
            perf_counter() - optimizer_step_start_time,
        )
        self._profile_add("train_step_seconds_total", perf_counter() - train_step_start_time)
        self._profile_add("train_step_count", 1)
        step_summary = self._summary_from_metrics(step_metrics, total_loss, total_items, stage=stage)
        if self.global_step % self.log_every_n_steps == 0:
            self._run_loggers("on_train_step", self._build_train_step_record(step_summary))
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
        path = self._resolve_artifact_path(path)
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

    def _run_epoch(self, data, stage, training, *, limit_override=None, val_data=None):
        batch_materialization_start_time = perf_counter()
        batches = list(self._batches(data))
        self._profile_add(
            "batch_materialization_seconds_total",
            perf_counter() - batch_materialization_start_time,
        )
        batches = self._apply_batch_limit(batches, stage=stage, override=limit_override)
        if not batches:
            raise ValueError(f"Trainer.{stage} requires at least one batch")

        if training and self.profiler == "simple":
            self._epoch_profile = self._empty_profile()
        self._active_stage = stage
        self._epoch_total_batches = len(batches)
        self._epoch_start_time = perf_counter()
        stage_start_time = self._epoch_start_time
        self._run_loggers(
            "on_stage_start",
            self._build_stage_start_record(stage, total_batches=len(batches)),
        )
        metrics = self._metrics()
        for metric in metrics:
            metric.reset()

        total_loss = 0.0
        total_items = 0
        self.model.train(training)
        context = torch.enable_grad() if training else torch.no_grad()
        mid_epoch_validation_points = (
            self._mid_epoch_validation_points(len(batches))
            if training and val_data is not None
            else set()
        )
        try:
            with context:
                if training:
                    self.optimizer.zero_grad()
                if training:
                    for group_start in range(0, len(batches), self.accumulate_grad_batches):
                        group_end = min(group_start + self.accumulate_grad_batches, len(batches))
                        self._active_batch_idx = group_end - 1
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
                        if group_end in mid_epoch_validation_points:
                            self._run_mid_epoch_validation(val_data)
                else:
                    for batch_index, batch in enumerate(batches):
                        self._active_batch_idx = batch_index
                        batch = self._prepare_batch(batch)
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
            return self._summary_from_metrics(metrics, total_loss, total_items, stage=stage)
        finally:
            self._profile_add(
                f"{stage}_stage_seconds_total",
                perf_counter() - stage_start_time,
            )
            self._active_batch_idx = None
            self._epoch_total_batches = None
            self._epoch_start_time = None

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
        if not self._checkpointing_enabled:
            return
        if self.save_best_path is None:
            return
        save_start_time = perf_counter()
        checkpoint_path = self.save_checkpoint(
            self.save_best_path,
            self.best_state_dict,
            metadata={
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "monitor": self.active_monitor,
            },
        )
        event_fields = checkpoint_event_fields(
            checkpoint_path,
            save_seconds=perf_counter() - save_start_time,
        )
        self.log_event(
            "checkpoint_saved",
            stage="fit",
            metrics=None if self.active_monitor is None or self.best_metric is None else {self.active_monitor: self.best_metric},
            checkpoint_tag="best",
            monitor_name=self.active_monitor,
            monitor_value=self.best_metric,
            **event_fields,
        )

    def evaluate(self, data, stage="val"):
        stage_start_time = perf_counter()
        summary = self._run_epoch(data, stage=stage, training=False)
        self._run_loggers(
            "on_evaluate_end",
            self._build_stage_record(
                summary,
                stage=stage,
                elapsed_seconds=perf_counter() - stage_start_time,
            ),
        )
        return summary

    def test(self, data):
        return self.evaluate(data, stage="test")

    def _uses_mid_epoch_validation(self):
        return not (isinstance(self.val_check_interval, float) and self.val_check_interval == 1.0)

    def _run_training_epoch(self, train_data, val_data=None):
        if val_data is None or not self._uses_mid_epoch_validation():
            return self._run_epoch(train_data, stage="train", training=True)
        return self._run_epoch(
            train_data,
            stage="train",
            training=True,
            val_data=val_data,
        )

    def _run_mid_epoch_validation(self, val_data):
        saved_stage = self._active_stage
        saved_total_batches = self._epoch_total_batches
        saved_epoch_start_time = self._epoch_start_time
        saved_batch_idx = self._active_batch_idx
        saved_training_mode = self.model.training
        stage_start_time = perf_counter()
        try:
            summary = self._run_epoch(val_data, stage="val", training=False)
        finally:
            self._active_stage = saved_stage
            self._epoch_total_batches = saved_total_batches
            self._epoch_start_time = saved_epoch_start_time
            self._active_batch_idx = saved_batch_idx
            self.model.train(saved_training_mode)
        self._run_loggers(
            "on_evaluate_end",
            self._build_stage_record(
                summary,
                stage="val",
                elapsed_seconds=perf_counter() - stage_start_time,
            ),
        )
        return summary

    def fit(self, train_data, val_data=None):
        monitor, mode = self._resolve_monitor(val_data)
        self._fit_epochs = self._effective_epochs()
        self._checkpointing_enabled = not self.fast_dev_run
        self._fit_profile = self._empty_profile()
        self._epoch_profile = None
        resume_state = self._resume_state
        history_kwargs = {
            "run_name": self.run_name,
            "root_dir": None if self.default_root_dir is None else str(self.default_root_dir),
            "fast_dev_run": self.fast_dev_run,
            "profiler": self.profiler,
        }
        if resume_state is None or resume_state.get("history_state") is None:
            history = TrainingHistory(epochs=self._fit_epochs, monitor=monitor, **history_kwargs)
            self.active_monitor = monitor
            self.global_step = 0
            start_epoch = 1
        else:
            history = TrainingHistory.from_state_dict(resume_state["history_state"])
            history["epochs"] = self._fit_epochs
            history["monitor"] = monitor
            history["stopped_early"] = False
            history["stop_reason"] = None
            history["run_name"] = history_kwargs["run_name"]
            history["root_dir"] = history_kwargs["root_dir"]
            history["fast_dev_run"] = history_kwargs["fast_dev_run"]
            history["sanity_check_passed"] = False
            history["profiler"] = history_kwargs["profiler"]
            history["profile"] = None
            start_epoch = int(history["completed_epochs"]) + 1
            trainer_state = resume_state.get("trainer_state", {})
            self.best_state_dict = trainer_state.get("best_state_dict")
            self.best_epoch = trainer_state.get("best_epoch")
            self.best_metric = trainer_state.get("best_metric")
            self.active_monitor = trainer_state.get("active_monitor", monitor) or monitor
            self.global_step = int(trainer_state.get("global_step", 0))
        self._fit_start_time = perf_counter()
        self._active_epoch = start_epoch - 1
        last_train_summary = None
        last_val_summary = None
        try:
            self._run_callbacks("on_fit_start", history=history)
            if resume_state is not None:
                self._restore_callback_states(resume_state.get("callback_states"))
                self._resume_state = None
            self._run_loggers("on_fit_start", self._build_fit_start_record(monitor))

            if val_data is not None and self.num_sanity_val_steps > 0:
                sanity_stage_start_time = perf_counter()
                sanity_summary = self._run_epoch(
                    val_data,
                    stage="sanity_val",
                    training=False,
                    limit_override=self.num_sanity_val_steps,
                )
                history["sanity_check_passed"] = True
                self._run_loggers(
                    "on_evaluate_end",
                    self._build_stage_record(
                        sanity_summary,
                        stage="sanity_val",
                        elapsed_seconds=perf_counter() - sanity_stage_start_time,
                    ),
                )

            for epoch in range(start_epoch, self._fit_epochs + 1):
                self._active_epoch = epoch
                epoch_start_time = perf_counter()
                train_summary = self._run_training_epoch(train_data, val_data=val_data)
                last_train_summary = train_summary
                val_summary = None
                if val_data is not None:
                    val_summary = self._run_epoch(val_data, stage="val", training=False)
                    last_val_summary = val_summary

                current = self._monitor_value(monitor, train_summary, val_summary)
                previous_best = self.best_metric
                improved = is_improvement(
                    current,
                    self.best_metric,
                    mode,
                )
                if improved:
                    improvement_delta = None
                    if previous_best is not None:
                        if mode == "min":
                            improvement_delta = float(previous_best - current)
                        else:
                            improvement_delta = float(current - previous_best)
                    self.best_metric = float(current)
                    self.best_epoch = epoch
                    self.best_state_dict = deepcopy(self.model.state_dict())
                    self.log_event(
                        "monitor_improved",
                        stage="fit",
                        metrics={monitor: float(current)},
                        monitor_name=monitor,
                        previous_best=None if previous_best is None else float(previous_best),
                        current_value=float(current),
                        improvement_delta=improvement_delta,
                    )
                    self._save_best()

                epoch_elapsed_seconds = perf_counter() - epoch_start_time
                history.record_epoch(
                    epoch=epoch,
                    train_summary=train_summary,
                    val_summary=val_summary,
                    best_epoch=self.best_epoch,
                    best_metric=self.best_metric,
                    elapsed_seconds=epoch_elapsed_seconds,
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
                    self._run_loggers(
                        "on_epoch_end",
                        self._build_epoch_record(
                            train_summary,
                            val_summary,
                            epoch_elapsed_seconds=epoch_elapsed_seconds,
                        ),
                    )
                    break
                self._run_loggers(
                    "on_epoch_end",
                    self._build_epoch_record(
                        train_summary,
                        val_summary,
                        epoch_elapsed_seconds=epoch_elapsed_seconds,
                    ),
                )
        except Exception as exc:
            self._run_callbacks("on_exception", exception=exc, history=history)
            self._run_loggers(
                "on_exception",
                self._build_exception_record(exc),
                suppress_errors=True,
            )
            self._finalize_loggers("exception", suppress_errors=True)
            self._fit_start_time = None
            raise

        history.finalize(
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
            final_train=last_train_summary,
            final_val=last_val_summary,
            fit_elapsed_seconds=self._fit_elapsed_seconds(),
            profile=self._profile_snapshot(self._fit_profile),
        )
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        self._run_callbacks("on_fit_end", history=history)
        try:
            self._run_loggers("on_fit_end", self._build_fit_end_record(history))
        finally:
            self._finalize_loggers("success", suppress_errors=True)
            self._fit_start_time = None
            self._epoch_profile = None
        self.last_history = history
        return history
