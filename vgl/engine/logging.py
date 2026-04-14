import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from vgl._optional import import_optional


class Logger:
    def on_fit_start(self, run_info):
        del run_info

    def on_stage_start(self, stage_record):
        del stage_record

    def on_train_step(self, log_record):
        del log_record

    def on_epoch_end(self, epoch_record):
        del epoch_record

    def on_evaluate_end(self, stage_record):
        del stage_record

    def on_fit_end(self, final_record):
        del final_record

    def on_exception(self, exception_record):
        del exception_record

    def on_event(self, record):
        del record

    def finalize(self, status):
        del status


def _normalize_events(events):
    if events is None:
        return None
    normalized = {str(event) for event in events}
    if not normalized:
        raise ValueError("events must not be empty")
    return normalized


def _normalize_metric_names(metric_names):
    if metric_names is None:
        return None
    normalized = {str(name) for name in metric_names}
    if not normalized:
        raise ValueError("metric_names must not be empty")
    return normalized


def _format_metric_value(value):
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def _format_size_bytes(size_bytes):
    if size_bytes is None:
        return "?"
    units = ("B", "KB", "MB", "GB")
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)}{units[unit_index]}"
    return f"{size:.1f}{units[unit_index]}"


def _format_duration(seconds):
    if seconds is None:
        return "?"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_metrics(metrics):
    if not metrics:
        return "-"
    return " ".join(f"{key}={_format_metric_value(metrics[key])}" for key in sorted(metrics))


_CORE_RECORD_FIELDS = (
    "event",
    "stage",
    "epoch",
    "epochs",
    "global_step",
    "batch_idx",
    "metrics",
)

_EVENT_CORE_RECORD_FIELDS = {
    "fit_start": (
        "monitor",
        "model_name",
        "task_name",
        "optimizer_name",
        "lr_scheduler_name",
        "precision",
        "total_parameters",
        "trainable_parameters",
    ),
    "exception": (
        "exception_type",
        "exception_message",
    ),
    "fit_end": (
        "best_epoch",
        "best_metric",
        "elapsed_seconds",
        "average_epoch_seconds",
        "average_steps_per_second",
        "stopped_early",
        "stop_reason",
    ),
    "monitor_improved": (
        "monitor_name",
        "previous_best",
        "current_value",
        "improvement_delta",
    ),
    "checkpoint_saved": (
        "checkpoint_tag",
        "path",
        "size_bytes",
        "save_seconds",
        "monitor_name",
        "monitor_value",
        "format",
        "format_version",
    ),
}


def _filter_metrics(metrics, *, metric_names, show_learning_rate):
    filtered = {}
    for metric_name, metric_value in metrics.items():
        if not show_learning_rate and (metric_name == "lr" or metric_name.startswith("lr/")):
            continue
        if metric_names is not None and metric_name not in metric_names:
            continue
        filtered[metric_name] = metric_value
    return filtered


def _filter_record(record, *, metric_names, include_context, show_learning_rate):
    filtered = {}
    core_fields = set(_CORE_RECORD_FIELDS)
    core_fields.update(_EVENT_CORE_RECORD_FIELDS.get(record.get("event"), ()))
    for key, value in record.items():
        if key == "metrics":
            filtered["metrics"] = _filter_metrics(
                value,
                metric_names=metric_names,
                show_learning_rate=show_learning_rate,
            )
            continue
        if include_context or key in core_fields:
            filtered[key] = value
    if "metrics" not in filtered:
        filtered["metrics"] = {}
    return filtered


def _tensorboard_summary_writer_class():
    return import_optional(
        "torch.utils.tensorboard",
        package_name="tensorboard",
        extra_name="tensorboard",
        feature_name="TensorBoardLogger",
    ).SummaryWriter


def _tensorboard_metric_name(metric_name, *, stage):
    if stage is not None:
        prefix = f"{stage}_"
        if metric_name.startswith(prefix):
            return metric_name[len(prefix) :]
    return metric_name


def _tensorboard_tag_prefix(record):
    event = record["event"]
    if event == "train_step":
        return "train_step"
    if event == "epoch_end":
        return "epoch"
    if event == "evaluate_end":
        return str(record.get("stage") or "evaluate")
    if event == "fit_end":
        return "fit"
    return f"events/{event}"


def _tensorboard_step(record):
    if record["event"] == "epoch_end":
        return int(record.get("epoch") or 0)
    return int(record.get("global_step") or 0)


class ConsoleLogger(Logger):
    _SUPPORTED_MODES = {"detailed", "compact"}
    _SUPPORTED_THEMES = {"default", "cat"}
    _CAT_FACES = {
        "starting": "(=^.^=)",
        "waiting": "(=^-^=)",
        "training": "(=^o^=)",
        "validating": "(=^?^=)",
        "testing": "(=^_^=)",
        "tracking": "(=^~^=)",
        "saving": "(=^v^=)",
        "done": "(=^.^=)~",
        "failed": "(=x.x=)",
    }

    def __init__(
        self,
        stream=None,
        enable_progress_bar=True,
        flush_every_n_steps=None,
        mode="detailed",
        metric_names=None,
        show_learning_rate=True,
        show_events=True,
        show_timestamp=True,
        theme="default",
    ):
        if flush_every_n_steps is not None and flush_every_n_steps < 1:
            raise ValueError("flush_every_n_steps must be >= 1")
        if mode not in self._SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {sorted(self._SUPPORTED_MODES)}")
        if theme not in self._SUPPORTED_THEMES:
            raise ValueError(f"theme must be one of {sorted(self._SUPPORTED_THEMES)}")
        self.stream = sys.stdout if stream is None else stream
        self.enable_progress_bar = bool(enable_progress_bar)
        self.flush_every_n_steps = flush_every_n_steps
        self.mode = mode
        self.metric_names = None if metric_names is None else {str(name) for name in metric_names}
        self.show_learning_rate = bool(show_learning_rate)
        self.show_events = bool(show_events)
        self.show_timestamp = bool(show_timestamp)
        self.theme = theme
        self._progress_active = False

    def _filtered_metrics(self, metrics):
        return _filter_metrics(
            metrics,
            metric_names=self.metric_names,
            show_learning_rate=self.show_learning_rate,
        )

    def _flush(self):
        flush = getattr(self.stream, "flush", None)
        if callable(flush):
            flush()

    def _prefix(self, *, status):
        parts = []
        if self.show_timestamp:
            parts.append(f"[{datetime.now().strftime('%H:%M:%S')}]")
        if self.theme == "cat":
            face = self._CAT_FACES.get(status, "(=^.^=)")
            parts.append(face)
            parts.append(f"{status} |")
        if not parts:
            return ""
        return " ".join(parts) + " "

    def _decorate_line(self, line, *, status):
        return f"{self._prefix(status=status)}{line}"

    def _write_line(self, line, *, status):
        if self._progress_active:
            self.stream.write("\n")
            self._progress_active = False
        self.stream.write(f"{self._decorate_line(line, status=status)}\n")
        self._flush()

    def _write_progress(self, line, *, step, status):
        self.stream.write(f"\r{self._decorate_line(line, status=status)}")
        self._progress_active = True
        if self.flush_every_n_steps is None or step % self.flush_every_n_steps == 0:
            self._flush()

    def _train_progress_bar(self, completed_batches, total_batches):
        if self.theme != "cat":
            return ""
        width = 10
        fraction = completed_batches / total_batches
        filled = min(width, max(0, int(round(width * fraction))))
        return f" | bar=[{'#' * filled}{'.' * (width - filled)}]"

    def _train_progress_fields(self, log_record):
        if self.mode != "detailed":
            return ""
        batch_idx = log_record.get("batch_idx")
        total_batches = log_record.get("epoch_total_batches")
        if batch_idx is None or total_batches in (None, 0):
            return ""
        completed_batches = min(int(batch_idx) + 1, int(total_batches))
        progress = 100.0 * completed_batches / int(total_batches)
        details = self._train_progress_bar(completed_batches, int(total_batches))
        details += f" | batch={completed_batches}/{int(total_batches)} ({progress:.1f}%)"
        epoch_elapsed_seconds = log_record.get("epoch_elapsed_seconds")
        if epoch_elapsed_seconds is not None and epoch_elapsed_seconds > 0 and completed_batches > 0:
            remaining_batches = max(int(total_batches) - completed_batches, 0)
            eta_seconds = remaining_batches * (float(epoch_elapsed_seconds) / completed_batches)
            details += f" | eta={_format_duration(eta_seconds)}"
        return details

    def _run_banner_lines(self, run_info):
        if self.mode != "detailed":
            return []
        lines = []
        summary_line = (
            f"model={run_info['model_name']}"
            f" | task={run_info['task_name']}"
            f" | optimizer={run_info['optimizer_name']}"
        )
        if run_info.get("lr_scheduler_name") is not None:
            summary_line += f" | scheduler={run_info['lr_scheduler_name']}"
        lines.extend(
            [
                "Run summary",
                summary_line,
                (
                    f"epochs={run_info['epochs']}"
                    f" | monitor={run_info['monitor']}"
                    f" | precision={run_info['precision']}"
                ),
                (
                    "parameters"
                    f" | total={run_info['total_parameters']}"
                    f" | trainable={run_info['trainable_parameters']}"
                ),
            ]
        )
        run_fields = []
        if run_info.get("run_name") is not None:
            run_fields.append(f"run_name={run_info['run_name']}")
        if run_info.get("root_dir") is not None:
            run_fields.append(f"root_dir={run_info['root_dir']}")
        if run_fields:
            lines.append("run | " + " | ".join(run_fields))
        controls = []
        if run_info.get("fast_dev_run"):
            controls.append("fast_dev_run=True")
        if int(run_info.get("num_sanity_val_steps") or 0) > 0:
            controls.append(f"sanity_val_steps={int(run_info['num_sanity_val_steps'])}")
        val_check_interval = run_info.get("val_check_interval")
        if not (isinstance(val_check_interval, float) and val_check_interval == 1.0):
            controls.append(f"val_check_interval={val_check_interval}")
        if controls:
            lines.append("controls | " + " | ".join(controls))
        lines.append(
            "batch_limits"
            f" | train={run_info.get('limit_train_batches')}"
            f" | val={run_info.get('limit_val_batches')}"
            f" | test={run_info.get('limit_test_batches')}"
        )
        return lines

    def _fit_progress_fields(self, *, epoch, epochs, elapsed_seconds=None, include_eta):
        if self.mode != "detailed" or epoch is None or epochs in (None, 0):
            return ""
        progress = 100.0 * int(epoch) / int(epochs)
        details = f" | fit={int(epoch)}/{int(epochs)} ({progress:.1f}%)"
        if include_eta and int(epoch) > 0 and elapsed_seconds is not None and int(epoch) < int(epochs):
            average_epoch_seconds = float(elapsed_seconds) / int(epoch)
            remaining_epochs = int(epochs) - int(epoch)
            details += f" | fit_eta={_format_duration(average_epoch_seconds * remaining_epochs)}"
        return details

    def on_fit_start(self, run_info):
        for line in self._run_banner_lines(run_info):
            self._write_line(line, status="starting")
        self._write_line(
            "Fit started"
            f" | epochs={run_info['epochs']}"
            f" | monitor={run_info['monitor']}"
            f" | precision={run_info['precision']}",
            status="starting",
        )

    def on_stage_start(self, stage_record):
        if self.mode != "detailed":
            return
        stage = stage_record["stage"]
        line = {
            "train": "Training started",
            "val": "Validation started",
            "sanity_val": "Sanity validation started",
            "test": "Test started",
        }.get(stage, f"{stage} started")
        epoch = stage_record.get("epoch")
        epochs = stage_record.get("epochs")
        if epoch is not None and epochs is not None and (epoch > 0 or stage == "sanity_val"):
            line += f" | epoch={epoch}/{epochs}"
        total_batches = stage_record.get("total_batches")
        if total_batches is not None:
            line += f" | batches={total_batches}"
        status = {
            "train": "training",
            "val": "validating",
            "sanity_val": "validating",
            "test": "testing",
        }.get(stage, "waiting")
        self._write_line(line, status=status)

    def on_train_step(self, log_record):
        metrics = self._filtered_metrics(log_record["metrics"])
        line = (
            f"Epoch {log_record['epoch']}/{log_record['epochs']}"
            f" | step={log_record['global_step']}"
            f" | {_format_metrics(metrics)}"
        )
        steps_per_second = log_record.get("steps_per_second") if self.mode == "detailed" else None
        if steps_per_second is not None:
            line += f" | steps_per_second={steps_per_second:.2f}"
        line += self._train_progress_fields(log_record)
        if self.enable_progress_bar:
            self._write_progress(line, step=log_record["global_step"], status="training")
        else:
            self._write_line(line, status="training")

    def on_epoch_end(self, epoch_record):
        metrics = self._filtered_metrics(epoch_record["metrics"])
        line = (
            f"Epoch {epoch_record['epoch']}/{epoch_record['epochs']}"
            f" | {_format_metrics(metrics)}"
        )
        if self.mode == "detailed":
            line += (
                f" | best_epoch={epoch_record['best_epoch']}"
                f" | best_metric={_format_metric_value(epoch_record['best_metric'])}"
                f" | global_step={epoch_record['global_step']}"
            )
            line += self._fit_progress_fields(
                epoch=epoch_record.get("epoch"),
                epochs=epoch_record.get("epochs"),
                elapsed_seconds=epoch_record.get("elapsed_seconds"),
                include_eta=True,
            )
        line += f" | epoch_time={epoch_record['elapsed_seconds']:.2f}s"
        self._write_line(line, status="training")

    def on_evaluate_end(self, stage_record):
        metrics = self._filtered_metrics(stage_record["metrics"])
        status = {
            "val": "validating",
            "sanity_val": "validating",
            "test": "testing",
        }.get(stage_record["stage"], "waiting")
        stage_label = {
            "sanity_val": "sanity validation summary",
        }.get(stage_record["stage"], f"{stage_record['stage']} summary")
        self._write_line(
            f"{stage_label}"
            f" | {_format_metrics(metrics)}"
            f" | elapsed={stage_record['elapsed_seconds']:.2f}s",
            status=status,
        )

    def on_fit_end(self, final_record):
        status = "stopped_early" if final_record.get("stopped_early") else "completed"
        metrics = self._filtered_metrics(final_record["metrics"])
        line = (
            "Fit complete"
            f" | status={status}"
            f" | {_format_metrics(metrics)}"
        )
        if self.mode == "detailed":
            line += (
                f" | best_epoch={final_record['best_epoch']}"
                f" | best_metric={_format_metric_value(final_record['best_metric'])}"
            )
            line += self._fit_progress_fields(
                epoch=final_record.get("epoch"),
                epochs=final_record.get("epochs"),
                elapsed_seconds=final_record.get("elapsed_seconds"),
                include_eta=False,
            )
            average_epoch_seconds = final_record.get("average_epoch_seconds")
            if average_epoch_seconds is not None:
                line += f" | avg_epoch_time={float(average_epoch_seconds):.2f}s"
            average_steps_per_second = final_record.get("average_steps_per_second")
            if average_steps_per_second is not None:
                line += f" | avg_steps_per_second={float(average_steps_per_second):.2f}"
        line += f" | elapsed={final_record['elapsed_seconds']:.2f}s"
        if final_record.get("stop_reason"):
            line += f" | stop_reason={final_record['stop_reason']}"
        self._write_line(line, status="done")

    def on_exception(self, exception_record):
        self._write_line(
            "Fit failed"
            f" | stage={exception_record['stage']}"
            f" | epoch={exception_record['epoch']}"
            f" | global_step={exception_record['global_step']}"
            f" | {exception_record['exception_type']}: {exception_record['exception_message']}",
            status="failed",
        )

    def on_event(self, record):
        if not self.show_events:
            return
        event = record["event"]
        if event == "monitor_improved":
            line = (
                "Monitor improved"
                f" | monitor={record['monitor_name']}"
                f" | value={_format_metric_value(record['current_value'])}"
                f" | previous_best={_format_metric_value(record['previous_best'])}"
            )
            if record.get("improvement_delta") is not None:
                line += f" | delta={_format_metric_value(record['improvement_delta'])}"
            line += f" | epoch={record['epoch']}"
            self._write_line(line, status="tracking")
            return
        if event == "checkpoint_saved":
            line = (
                "Checkpoint saved"
                f" | tag={record['checkpoint_tag']}"
                f" | path={record['path']}"
            )
            if record.get("monitor_name") is not None:
                line += (
                    f" | monitor={record['monitor_name']}"
                    f" | value={_format_metric_value(record.get('monitor_value'))}"
                )
            if record.get("size_bytes") is not None:
                line += f" | size={_format_size_bytes(record.get('size_bytes'))}"
            if record.get("save_seconds") is not None:
                line += f" | save_time={float(record['save_seconds']):.2f}s"
            self._write_line(line, status="saving")
            return
        self._write_line(
            f"Event {event} | {_format_metrics(self._filtered_metrics(record.get('metrics', {})))}",
            status="event",
        )

    def finalize(self, status):
        del status
        if self._progress_active:
            self.stream.write("\n")
            self._progress_active = False
            self._flush()


class JSONLinesLogger(Logger):
    def __init__(
        self,
        path,
        *,
        flush=False,
        events=None,
        metric_names=None,
        include_context=True,
        show_learning_rate=True,
    ):
        self.path = Path(path)
        self.flush = bool(flush)
        self.events = _normalize_events(events)
        self.metric_names = _normalize_metric_names(metric_names)
        self.include_context = bool(include_context)
        self.show_learning_rate = bool(show_learning_rate)
        self._handle = None

    def _ensure_handle(self):
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle

    def _write(self, record):
        if self.events is not None and record["event"] not in self.events:
            return
        record = _filter_record(
            record,
            metric_names=self.metric_names,
            include_context=self.include_context,
            show_learning_rate=self.show_learning_rate,
        )
        handle = self._ensure_handle()
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.flush:
            handle.flush()

    def on_fit_start(self, run_info):
        self._write(run_info)

    def on_train_step(self, log_record):
        self._write(log_record)

    def on_epoch_end(self, epoch_record):
        self._write(epoch_record)

    def on_evaluate_end(self, stage_record):
        self._write(stage_record)

    def on_fit_end(self, final_record):
        self._write(final_record)

    def on_exception(self, exception_record):
        self._write(exception_record)

    def on_event(self, record):
        self._write(record)

    def finalize(self, status):
        del status
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class CSVLogger(Logger):
    def __init__(
        self,
        path,
        *,
        flush=False,
        events=None,
        metric_names=None,
        include_context=True,
        show_learning_rate=True,
    ):
        self.path = Path(path)
        self.flush = bool(flush)
        self.events = _normalize_events({"epoch_end"} if events is None else events)
        self.metric_names = _normalize_metric_names(metric_names)
        self.include_context = bool(include_context)
        self.show_learning_rate = bool(show_learning_rate)
        self._rows = []
        self._fieldnames = []

    def _flatten_record(self, record):
        flat = {}
        for key, value in record.items():
            if key == "metrics":
                for metric_name, metric_value in value.items():
                    flat[metric_name] = metric_value
                continue
            if isinstance(value, (dict, list, tuple)):
                flat[key] = json.dumps(value, sort_keys=True)
            else:
                flat[key] = value
        return flat

    def _write(self, record):
        if record["event"] not in self.events:
            return
        record = _filter_record(
            record,
            metric_names=self.metric_names,
            include_context=self.include_context,
            show_learning_rate=self.show_learning_rate,
        )
        row = self._flatten_record(record)
        self._rows.append(row)
        fieldnames = set(self._fieldnames)
        fieldnames.update(row)
        preferred_prefix = [
            "event",
            "stage",
            "epoch",
            "epochs",
            "global_step",
            "batch_idx",
            "monitor",
            "best_epoch",
            "best_metric",
            "elapsed_seconds",
        ]
        ordered = [name for name in preferred_prefix if name in fieldnames]
        ordered.extend(sorted(fieldnames - set(ordered)))
        self._fieldnames = ordered
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()
            for existing_row in self._rows:
                writer.writerow(existing_row)
            if self.flush:
                handle.flush()

    def on_fit_start(self, run_info):
        self._write(run_info)

    def on_train_step(self, log_record):
        self._write(log_record)

    def on_epoch_end(self, epoch_record):
        self._write(epoch_record)

    def on_evaluate_end(self, stage_record):
        self._write(stage_record)

    def on_fit_end(self, final_record):
        self._write(final_record)

    def on_exception(self, exception_record):
        self._write(exception_record)

    def on_event(self, record):
        self._write(record)


class TensorBoardLogger(Logger):
    def __init__(
        self,
        log_dir,
        *,
        flush=False,
        events=None,
        metric_names=None,
        show_learning_rate=True,
    ):
        self.log_dir = Path(log_dir)
        self.flush = bool(flush)
        self.events = _normalize_events(events)
        self.metric_names = _normalize_metric_names(metric_names)
        self.show_learning_rate = bool(show_learning_rate)
        self._writer = None

    def _ensure_writer(self):
        if self._writer is None:
            summary_writer = _tensorboard_summary_writer_class()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = summary_writer(log_dir=str(self.log_dir))
        return self._writer

    def _should_log_event(self, event):
        return self.events is None or event in self.events

    def _filtered_metrics(self, metrics):
        return _filter_metrics(
            metrics,
            metric_names=self.metric_names,
            show_learning_rate=self.show_learning_rate,
        )

    def _flush_if_needed(self):
        if self.flush and self._writer is not None:
            self._writer.flush()

    def _write_scalars(self, record, *, tag_prefix=None):
        if not self._should_log_event(record["event"]):
            return
        metrics = self._filtered_metrics(record.get("metrics", {}))
        if not metrics:
            return
        writer = self._ensure_writer()
        step = _tensorboard_step(record)
        prefix = _tensorboard_tag_prefix(record) if tag_prefix is None else tag_prefix
        stage = record.get("stage")
        for metric_name, metric_value in sorted(metrics.items()):
            writer.add_scalar(
                f"{prefix}/{_tensorboard_metric_name(metric_name, stage=stage)}",
                float(metric_value),
                step,
            )
        self._flush_if_needed()

    def on_fit_start(self, run_info):
        if not self._should_log_event(run_info["event"]):
            return
        metadata = {key: value for key, value in run_info.items() if key != "metrics"}
        writer = self._ensure_writer()
        writer.add_text("run/metadata", json.dumps(metadata, sort_keys=True), 0)
        self._flush_if_needed()

    def on_train_step(self, log_record):
        self._write_scalars(log_record)

    def on_epoch_end(self, epoch_record):
        self._write_scalars(epoch_record)

    def on_evaluate_end(self, stage_record):
        self._write_scalars(stage_record)

    def on_fit_end(self, final_record):
        self._write_scalars(final_record)

    def on_exception(self, exception_record):
        if not self._should_log_event(exception_record["event"]):
            return
        writer = self._ensure_writer()
        writer.add_text(
            "run/exception",
            json.dumps(exception_record, sort_keys=True),
            _tensorboard_step(exception_record),
        )
        self._flush_if_needed()

    def on_event(self, record):
        if not self._should_log_event(record["event"]):
            return
        event = record["event"]
        writer = self._ensure_writer()
        step = _tensorboard_step(record)
        if event == "monitor_improved":
            monitor_name = str(record["monitor_name"])
            writer.add_scalar(f"monitor/{monitor_name}", float(record["current_value"]), step)
            previous_best = record.get("previous_best")
            if previous_best is not None:
                writer.add_scalar(f"monitor/{monitor_name}_previous_best", float(previous_best), step)
            improvement_delta = record.get("improvement_delta")
            if improvement_delta is not None:
                writer.add_scalar(f"monitor/{monitor_name}_delta", float(improvement_delta), step)
            self._flush_if_needed()
            return
        if event == "checkpoint_saved":
            writer.add_text(f"checkpoint/{record['checkpoint_tag']}", str(record["path"]), step)
            size_bytes = record.get("size_bytes")
            if size_bytes is not None:
                writer.add_scalar(f"checkpoint/{record['checkpoint_tag']}_size_bytes", float(size_bytes), step)
            save_seconds = record.get("save_seconds")
            if save_seconds is not None:
                writer.add_scalar(f"checkpoint/{record['checkpoint_tag']}_save_seconds", float(save_seconds), step)
            self._flush_if_needed()
            return
        self._write_scalars(record)

    def finalize(self, status):
        del status
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
