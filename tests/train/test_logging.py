import csv
import json
import re

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from vgl.engine import (
    Callback,
    CSVLogger,
    ConsoleLogger,
    JSONLinesLogger,
    Logger,
    ModelCheckpoint,
    TensorBoardLogger,
    Trainer,
)
from vgl.train.task import Task


class ToyBatch:
    def __init__(self, target):
        self.target = torch.tensor([target], dtype=torch.float32)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, batch):
        return self.weight.repeat(batch.target.size(0))


class MultiGroupToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_a = nn.Parameter(torch.tensor([0.0]))
        self.weight_b = nn.Parameter(torch.tensor([0.0]))

    def forward(self, batch):
        return (self.weight_a + self.weight_b).repeat(batch.target.size(0))


class ToyTask(Task):
    def loss(self, batch, predictions, stage):
        del stage
        return ((predictions - batch.target) ** 2).mean()

    def targets(self, batch, stage):
        del stage
        return batch.target


class RecordingLogger(Logger):
    def __init__(self):
        self.events = []

    def on_fit_start(self, run_info):
        self.events.append(dict(run_info))

    def on_train_step(self, log_record):
        self.events.append(dict(log_record))

    def on_epoch_end(self, epoch_record):
        self.events.append(dict(epoch_record))

    def on_evaluate_end(self, stage_record):
        self.events.append(dict(stage_record))

    def on_fit_end(self, final_record):
        self.events.append(dict(final_record))

    def on_exception(self, exception_record):
        self.events.append(dict(exception_record))

    def on_event(self, record):
        self.events.append(dict(record))


class FinalizeRecordingLogger(Logger):
    def __init__(self):
        self.finalize_statuses = []
        self.exception_events = []

    def on_exception(self, exception_record):
        self.exception_events.append(dict(exception_record))

    def finalize(self, status):
        self.finalize_statuses.append(status)


class RaiseOnFitEndLogger(FinalizeRecordingLogger):
    def on_fit_end(self, final_record):
        del final_record
        raise RuntimeError("fit-end logger boom")


class RaiseOnExceptionLogger(FinalizeRecordingLogger):
    def on_exception(self, exception_record):
        super().on_exception(exception_record)
        raise RuntimeError("exception logger boom")


class RaiseOnEpochEndCallback(Callback):
    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, epoch, train_summary, val_summary, history
        raise RuntimeError("callback boom")


class RaiseOnFirstEpochCallback(Callback):
    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary, history
        if epoch == 1:
            raise RuntimeError("boom")


def _load_event_accumulator(log_dir):
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return accumulator


def test_trainer_adds_default_console_logger_when_enabled():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    assert any(isinstance(logger, ConsoleLogger) for logger in trainer.loggers)


def test_trainer_emits_logger_events_on_optimizer_steps():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[logger],
        log_every_n_steps=2,
        enable_console_logging=False,
        accumulate_grad_batches=2,
    )

    history = trainer.fit([ToyBatch(1.0) for _ in range(6)])

    assert history["completed_epochs"] == 1
    events = [event["event"] for event in logger.events]
    assert events[0] == "fit_start"
    assert "train_step" in events
    assert "epoch_end" in events
    assert events[-1] == "fit_end"
    train_step = next(event for event in logger.events if event["event"] == "train_step")
    epoch_end = next(event for event in logger.events if event["event"] == "epoch_end")
    fit_end = logger.events[-1]
    assert logger.events[0]["monitor"] == "train_loss"
    assert train_step["stage"] == "train"
    assert train_step["epoch"] == 1
    assert train_step["global_step"] == 2
    assert train_step["metrics"]["loss"] > 0.0
    assert epoch_end["metrics"]["train_loss"] > 0.0
    assert fit_end["global_step"] == 3
    assert fit_end["best_epoch"] == 1
    assert fit_end["average_epoch_seconds"] > 0.0
    assert fit_end["average_steps_per_second"] > 0.0


def test_fit_start_record_includes_run_metadata():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[logger],
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)])

    fit_start = logger.events[0]

    assert fit_start["event"] == "fit_start"
    assert fit_start["model_name"] == "ToyModel"
    assert fit_start["task_name"] == "ToyTask"
    assert fit_start["optimizer_name"] == "SGD"
    assert fit_start["lr_scheduler_name"] is None
    assert fit_start["total_parameters"] == 1
    assert fit_start["trainable_parameters"] == 1
    assert fit_start["callback_names"] == []
    assert fit_start["logger_names"] == ["RecordingLogger"]


def test_trainer_finalizes_loggers_when_fit_end_logger_raises():
    finalize_logger = FinalizeRecordingLogger()
    raising_logger = RaiseOnFitEndLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[finalize_logger, raising_logger],
        enable_console_logging=False,
    )

    with pytest.raises(RuntimeError, match="fit-end logger boom"):
        trainer.fit([ToyBatch(1.0)])

    assert finalize_logger.finalize_statuses == ["success"]
    assert raising_logger.finalize_statuses == ["success"]


def test_trainer_finalizes_loggers_when_exception_logger_raises():
    finalize_logger = FinalizeRecordingLogger()
    raising_logger = RaiseOnExceptionLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        callbacks=[RaiseOnEpochEndCallback()],
        loggers=[finalize_logger, raising_logger],
        enable_console_logging=False,
    )

    with pytest.raises(RuntimeError, match="callback boom"):
        trainer.fit([ToyBatch(1.0)])

    assert finalize_logger.finalize_statuses == ["exception"]
    assert raising_logger.finalize_statuses == ["exception"]
    assert finalize_logger.exception_events[0]["exception_type"] == "RuntimeError"


def test_fit_start_and_checkpoint_records_include_artifact_paths(tmp_path):
    logger = RecordingLogger()
    root_dir = tmp_path / "artifacts"
    checkpoint_callback = ModelCheckpoint("checkpoints", save_top_k=1, save_last=True)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        callbacks=[checkpoint_callback],
        loggers=[logger, JSONLinesLogger("logs/train.jsonl", flush=True)],
        enable_console_logging=False,
        default_root_dir=root_dir,
        run_name="artifact-smoke",
        save_best_path="best.pt",
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    fit_start = logger.events[0]
    checkpoint_event = next(
        event
        for event in logger.events
        if event["event"] == "checkpoint_saved" and event["checkpoint_tag"] == "best"
    )

    assert fit_start["run_name"] == "artifact-smoke"
    assert fit_start["artifact_paths"] == {
        "save_best_path": str(root_dir / "best.pt"),
        "logger_paths": [
            {
                "logger": "JSONLinesLogger",
                "path": str(root_dir / "logs" / "train.jsonl"),
            }
        ],
        "callback_dirs": [
            {
                "callback": "ModelCheckpoint",
                "dirpath": str(root_dir / "checkpoints"),
            }
        ],
    }
    assert checkpoint_event["artifact_paths"] == fit_start["artifact_paths"]


def test_trainer_emits_evaluate_end_records_for_evaluate_and_test():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[logger],
        enable_console_logging=False,
    )

    val_summary = trainer.evaluate([ToyBatch(1.0)])
    test_summary = trainer.test([ToyBatch(2.0)])

    assert val_summary["loss"] > 0.0
    assert test_summary["loss"] > 0.0
    assert [event["event"] for event in logger.events] == ["evaluate_end", "evaluate_end"]
    assert [event["stage"] for event in logger.events] == ["val", "test"]
    assert logger.events[0]["metrics"]["val_loss"] == val_summary["loss"]
    assert logger.events[1]["metrics"]["test_loss"] == test_summary["loss"]


def test_trainer_emits_monitor_improved_and_best_checkpoint_events(tmp_path):
    logger = RecordingLogger()
    checkpoint = tmp_path / "best.pt"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=2,
        save_best_path=checkpoint,
        loggers=[logger],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    improved_events = [event for event in logger.events if event["event"] == "monitor_improved"]
    checkpoint_events = [
        event
        for event in logger.events
        if event["event"] == "checkpoint_saved" and event["checkpoint_tag"] == "best"
    ]

    assert len(improved_events) == 2
    assert improved_events[0]["monitor_name"] == "val_loss"
    assert improved_events[0]["previous_best"] is None
    assert improved_events[0]["improvement_delta"] is None
    assert improved_events[1]["previous_best"] == improved_events[0]["current_value"]
    assert improved_events[1]["improvement_delta"] == (
        improved_events[0]["current_value"] - improved_events[1]["current_value"]
    )
    assert len(checkpoint_events) == 2
    assert checkpoint_events[-1]["path"] == str(checkpoint)
    assert checkpoint_events[-1]["monitor_name"] == "val_loss"
    assert checkpoint_events[-1]["size_bytes"] > 0
    assert checkpoint_events[-1]["save_seconds"] >= 0.0


def test_model_checkpoint_callback_emits_checkpoint_saved_events(tmp_path):
    logger = RecordingLogger()
    callback = ModelCheckpoint(tmp_path, save_top_k=1, save_last=True)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        callbacks=[callback],
        loggers=[logger],
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    checkpoint_events = [event for event in logger.events if event["event"] == "checkpoint_saved"]

    assert [event["checkpoint_tag"] for event in checkpoint_events] == ["last", "top_k"]
    assert checkpoint_events[0]["path"].endswith("last.ckpt")
    assert checkpoint_events[1]["path"].endswith(".ckpt")
    assert checkpoint_events[1]["monitor_name"] == "val_loss"
    assert checkpoint_events[0]["size_bytes"] > 0
    assert checkpoint_events[0]["save_seconds"] >= 0.0
    assert checkpoint_events[1]["size_bytes"] > 0
    assert checkpoint_events[1]["save_seconds"] >= 0.0


def test_json_lines_logger_writes_structured_fit_events(tmp_path):
    path = tmp_path / "training.jsonl"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[JSONLinesLogger(path, flush=True)],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0), ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert [record["event"] for record in records] == [
        "fit_start",
        "train_step",
        "train_step",
        "monitor_improved",
        "epoch_end",
        "fit_end",
    ]
    assert records[0]["monitor"] == "val_loss"
    assert records[1]["global_step"] == 1
    assert records[2]["global_step"] == 2
    assert records[3]["monitor_name"] == "val_loss"
    assert records[4]["metrics"]["val_loss"] > 0.0
    assert records[5]["best_epoch"] == 1


def test_json_lines_logger_can_filter_to_epoch_events_only(tmp_path):
    path = tmp_path / "epoch_only.jsonl"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[JSONLinesLogger(path, flush=True, events={"epoch_end", "fit_end"})],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0), ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert [record["event"] for record in records] == ["epoch_end", "fit_end"]
    assert "train_loss" in records[0]["metrics"]
    assert records[1]["best_epoch"] == 1


def test_json_lines_logger_can_filter_metrics_and_drop_context(tmp_path):
    path = tmp_path / "filtered.jsonl"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[
            JSONLinesLogger(
                path,
                flush=True,
                events={"epoch_end", "fit_end"},
                metric_names={"train_loss", "val_loss"},
                include_context=False,
            )
        ],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0), ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert records[0].keys() == {
        "event",
        "stage",
        "epoch",
        "epochs",
        "global_step",
        "batch_idx",
        "metrics",
    }
    assert set(records[0]["metrics"]) == {"train_loss", "val_loss"}
    assert "lr" not in records[0]["metrics"]
    assert records[1].keys() == {
        "event",
        "stage",
        "epoch",
        "epochs",
        "global_step",
        "batch_idx",
        "metrics",
        "best_epoch",
        "best_metric",
        "elapsed_seconds",
        "average_epoch_seconds",
        "average_steps_per_second",
        "stopped_early",
        "stop_reason",
    }


def test_json_lines_logger_preserves_fit_end_core_fields_without_context(tmp_path):
    path = tmp_path / "fit_end.jsonl"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[
            JSONLinesLogger(
                path,
                flush=True,
                events={"fit_end"},
                include_context=False,
            )
        ],
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(records) == 1
    assert records[0].keys() == {
        "average_epoch_seconds",
        "average_steps_per_second",
        "batch_idx",
        "best_epoch",
        "best_metric",
        "elapsed_seconds",
        "epoch",
        "epochs",
        "event",
        "global_step",
        "metrics",
        "stage",
        "stop_reason",
        "stopped_early",
    }
    assert records[0]["event"] == "fit_end"
    assert records[0]["best_epoch"] == 1
    assert records[0]["stop_reason"] is None
    assert records[0]["stopped_early"] is False


def test_json_lines_logger_preserves_fit_start_core_fields_without_context(tmp_path):
    path = tmp_path / "fit_start.jsonl"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[
            JSONLinesLogger(
                path,
                flush=True,
                events={"fit_start"},
                include_context=False,
            )
        ],
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)])

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(records) == 1
    assert records[0].keys() == {
        "batch_idx",
        "epoch",
        "epochs",
        "event",
        "global_step",
        "metrics",
        "monitor",
        "model_name",
        "task_name",
        "optimizer_name",
        "lr_scheduler_name",
        "precision",
        "stage",
        "total_parameters",
        "trainable_parameters",
    }
    assert records[0]["event"] == "fit_start"
    assert records[0]["monitor"] == "train_loss"
    assert records[0]["model_name"] == "ToyModel"
    assert records[0]["task_name"] == "ToyTask"
    assert records[0]["optimizer_name"] == "SGD"
    assert records[0]["metrics"] == {}
    assert "callback_names" not in records[0]
    assert "logger_names" not in records[0]


def test_csv_logger_preserves_exception_core_fields_without_context(tmp_path):
    path = tmp_path / "exception.csv"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        callbacks=[RaiseOnFirstEpochCallback()],
        loggers=[
            CSVLogger(
                path,
                flush=True,
                events={"exception"},
                include_context=False,
            )
        ],
        enable_console_logging=False,
    )

    with pytest.raises(RuntimeError, match="boom"):
        trainer.fit([ToyBatch(1.0)])

    rows = list(csv.DictReader(path.read_text().splitlines()))

    assert len(rows) == 1
    assert set(rows[0]) == {
        "event",
        "stage",
        "epoch",
        "epochs",
        "global_step",
        "batch_idx",
        "exception_type",
        "exception_message",
    }
    assert rows[0]["event"] == "exception"
    assert rows[0]["exception_type"] == "RuntimeError"
    assert rows[0]["exception_message"] == "boom"


def test_json_lines_logger_can_hide_learning_rate_metrics(tmp_path):
    path = tmp_path / "no_lr.jsonl"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[
            JSONLinesLogger(
                path,
                flush=True,
                events={"epoch_end", "fit_end"},
                show_learning_rate=False,
            )
        ],
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(records) == 2
    assert {"train_loss", "val_loss"} <= set(records[0]["metrics"])
    assert all("lr" not in record["metrics"] for record in records)


def test_csv_logger_writes_epoch_summary_rows(tmp_path):
    path = tmp_path / "training.csv"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=2,
        loggers=[CSVLogger(path, flush=True)],
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    rows = list(csv.DictReader(path.read_text().splitlines()))

    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert rows[0]["event"] == "epoch_end"
    assert rows[0]["train_loss"] != ""
    assert rows[0]["val_loss"] != ""
    assert rows[0]["lr"] == "0.1"
    assert rows[1]["epoch"] == "2"


def test_csv_logger_can_filter_metrics_and_drop_context_columns(tmp_path):
    path = tmp_path / "filtered.csv"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[
            CSVLogger(
                path,
                flush=True,
                events={"epoch_end"},
                metric_names={"train_loss", "val_loss"},
                include_context=False,
            )
        ],
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    rows = list(csv.DictReader(path.read_text().splitlines()))

    assert len(rows) == 1
    assert set(rows[0]) == {
        "event",
        "stage",
        "epoch",
        "epochs",
        "global_step",
        "batch_idx",
        "train_loss",
        "val_loss",
    }
    assert "lr" not in rows[0]
    assert "best_epoch" not in rows[0]


def test_csv_logger_can_hide_multi_group_learning_rate_columns(tmp_path):
    path = tmp_path / "no_group_lr.csv"
    model = MultiGroupToyModel()
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        optimizer_param_groups=[
            {"params": [model.weight_a], "lr": 0.1},
            {"params": [model.weight_b], "lr": 0.01},
        ],
        loggers=[CSVLogger(path, flush=True, show_learning_rate=False)],
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    rows = list(csv.DictReader(path.read_text().splitlines()))

    assert len(rows) == 1
    assert rows[0]["train_loss"] != ""
    assert rows[0]["val_loss"] != ""
    assert not any(key == "lr" or key.startswith("lr/") for key in rows[0])


def test_tensorboard_logger_writes_scalars_and_metadata(tmp_path):
    log_dir = tmp_path / "tensorboard"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[TensorBoardLogger(log_dir, flush=True)],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0), ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    accumulator = _load_event_accumulator(log_dir)
    scalar_tags = set(accumulator.Tags()["scalars"])
    tensor_tags = set(accumulator.Tags()["tensors"])

    assert "run/metadata/text_summary" in tensor_tags
    assert "train_step/loss" in scalar_tags
    assert "train_step/lr" in scalar_tags
    assert "epoch/train_loss" in scalar_tags
    assert "epoch/val_loss" in scalar_tags
    assert "fit/train_loss" in scalar_tags
    assert "fit/val_loss" in scalar_tags
    assert [event.step for event in accumulator.Scalars("train_step/loss")] == [1, 2]
    assert [event.step for event in accumulator.Scalars("epoch/train_loss")] == [1]
    assert [event.step for event in accumulator.Scalars("fit/train_loss")] == [2]


def test_tensorboard_logger_can_filter_events_metrics_and_learning_rate(tmp_path):
    log_dir = tmp_path / "tensorboard-filtered"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[
            TensorBoardLogger(
                log_dir,
                flush=True,
                events={"epoch_end", "fit_end"},
                metric_names={"train_loss", "val_loss"},
                show_learning_rate=False,
            )
        ],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0), ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    accumulator = _load_event_accumulator(log_dir)
    scalar_tags = set(accumulator.Tags()["scalars"])
    tensor_tags = set(accumulator.Tags()["tensors"])

    assert tensor_tags == set()
    assert scalar_tags == {
        "epoch/train_loss",
        "epoch/val_loss",
        "fit/train_loss",
        "fit/val_loss",
    }


def test_console_logger_prints_epoch_and_fit_summaries(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[ConsoleLogger()],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "Fit started" in output
    assert "Epoch 1/1" in output
    assert "train_loss" in output
    assert "val_loss" in output
    assert "Fit complete" in output
    assert "avg_epoch_time=" in output
    assert "avg_steps_per_second=" in output


def test_console_logger_detailed_mode_prints_run_banner_before_fit_start(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[ConsoleLogger(enable_progress_bar=False)],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "Run summary" in output
    assert "model=ToyModel | task=ToyTask | optimizer=SGD" in output
    assert "epochs=1 | monitor=train_loss | precision=32" in output
    assert "parameters | total=1 | trainable=1" in output
    assert output.index("Run summary") < output.index("Fit started")


def test_console_logger_detailed_mode_prints_run_controls_in_banner(capsys, tmp_path):
    root_dir = tmp_path / "artifacts"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=3,
        loggers=[ConsoleLogger(enable_progress_bar=False)],
        enable_console_logging=False,
        default_root_dir=root_dir,
        run_name="smoke-run",
        fast_dev_run=True,
        limit_train_batches=0.5,
        limit_val_batches=0.25,
        limit_test_batches=2,
        num_sanity_val_steps=1,
        val_check_interval=0.5,
    )

    trainer.fit([ToyBatch(1.0) for _ in range(4)], val_data=[ToyBatch(1.0) for _ in range(4)])

    output = capsys.readouterr().out

    assert "run | run_name=smoke-run" in output
    assert f"root_dir={root_dir}" in output
    assert "controls | fast_dev_run=True | sanity_val_steps=1 | val_check_interval=0.5" in output
    assert "batch_limits | train=0.5 | val=0.25 | test=2" in output


def test_console_logger_can_show_timestamp_and_cat_statuses(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[
            ConsoleLogger(
                enable_progress_bar=False,
                show_timestamp=True,
                theme="cat",
            )
        ],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])
    trainer.evaluate([ToyBatch(1.0)])
    trainer.test([ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] \(=\^\.\^=\) starting \| Fit started", output)
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] \(=\^o\^=\) training \| Epoch 1/1", output)
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] \(=\^\?\^=\) validating \| val summary", output)
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] \(=\^\.\^=\)~ done \| Fit complete", output)
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] \(=\^_\^=\) testing \| test summary", output)


def test_console_logger_detailed_mode_shows_tqdm_like_batch_progress_and_eta(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[ConsoleLogger(enable_progress_bar=False)],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0) for _ in range(4)])

    output = capsys.readouterr().out

    assert "batch=1/4 (25.0%)" in output
    assert "batch=4/4 (100.0%)" in output
    assert "eta=" in output
    assert "steps_per_second=" in output


def test_console_logger_detailed_mode_shows_fit_progress_and_fit_eta(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=2,
        loggers=[ConsoleLogger(enable_progress_bar=False)],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "fit=1/2 (50.0%)" in output
    assert "fit=2/2 (100.0%)" in output
    assert "fit_eta=" in output


def test_console_logger_detailed_mode_prints_stage_start_markers(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[ConsoleLogger(enable_progress_bar=False)],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])
    trainer.test([ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "Training started | epoch=1/1 | batches=1" in output
    assert "Validation started | epoch=1/1 | batches=1" in output
    assert "Test started | epoch=1/1 | batches=1" in output


def test_console_logger_detailed_mode_shows_sanity_validation_stage(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[ConsoleLogger(enable_progress_bar=False)],
        enable_console_logging=False,
        num_sanity_val_steps=1,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "Sanity validation started | epoch=0/1 | batches=1" in output
    assert "sanity validation summary" in output


def test_console_logger_cat_theme_shows_stage_start_statuses(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[ConsoleLogger(enable_progress_bar=False, theme="cat")],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])
    trainer.test([ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "(=^o^=) training | Training started | epoch=1/1 | batches=1" in output
    assert "(=^?^=) validating | Validation started | epoch=1/1 | batches=1" in output
    assert "(=^_^=) testing | Test started | epoch=1/1 | batches=1" in output


def test_console_logger_shows_monitor_improvement_delta(capsys, tmp_path):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=2,
        save_best_path=tmp_path / "best.pt",
        loggers=[ConsoleLogger(enable_progress_bar=False)],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "Monitor improved" in output
    assert "delta=" in output


def test_console_logger_shows_checkpoint_size_and_save_time(capsys, tmp_path):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        save_best_path=tmp_path / "best.pt",
        loggers=[ConsoleLogger(enable_progress_bar=False)],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "Checkpoint saved" in output
    assert "size=" in output
    assert "save_time=" in output


def test_console_logger_cat_theme_shows_ascii_progress_bar(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[ConsoleLogger(enable_progress_bar=False, theme="cat")],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0) for _ in range(4)])

    output = capsys.readouterr().out

    assert "bar=[##........]" in output
    assert "bar=[##########]" in output


def test_console_logger_compact_mode_can_filter_metrics_and_hide_events(capsys, tmp_path):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=2,
        loggers=[
            ConsoleLogger(
                enable_progress_bar=False,
                mode="compact",
                metric_names={"loss", "train_loss", "val_loss"},
                show_learning_rate=False,
                show_events=False,
            )
        ],
        log_every_n_steps=1,
        save_best_path=tmp_path / "best.pt",
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "Fit started" in output
    assert "Epoch 1/2" in output
    assert "train_loss" in output
    assert "val_loss" in output
    assert "lr=" not in output
    assert "steps_per_second" not in output
    assert "bar=" not in output
    assert "batch=" not in output
    assert "eta=" not in output
    assert "fit=" not in output
    assert "fit_eta=" not in output
    assert "avg_epoch_time=" not in output
    assert "avg_steps_per_second=" not in output
    assert "best_epoch" not in output
    assert "Monitor improved" not in output
    assert "Checkpoint saved" not in output


def test_console_logger_compact_mode_skips_run_banner(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[ConsoleLogger(enable_progress_bar=False, mode="compact")],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "Fit started" in output
    assert "Run summary" not in output
    assert "model=ToyModel | task=ToyTask | optimizer=SGD" not in output
    assert "parameters | total=1 | trainable=1" not in output
    assert "Training started" not in output
    assert "Validation started" not in output


def test_trainer_default_console_logger_supports_timestamp_and_cat_theme(capsys, tmp_path):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        log_every_n_steps=1,
        save_best_path=tmp_path / "best.pt",
        enable_progress_bar=False,
        console_show_timestamp=True,
        console_theme="cat",
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert re.search(r"\[\d{2}:\d{2}:\d{2}\]", output)
    assert "(=^.^=) starting | Fit started" in output
    assert "(=^o^=) training | Epoch 1/1" in output
    assert "(=^~^=) tracking | Monitor improved" in output
    assert "(=^v^=) saving | Checkpoint saved" in output
    assert "(=^.^=)~ done | Fit complete" in output


def test_trainer_default_console_logger_respects_console_configuration(capsys, tmp_path):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        log_every_n_steps=1,
        save_best_path=tmp_path / "best.pt",
        enable_progress_bar=False,
        console_mode="compact",
        console_metric_names={"loss", "train_loss", "val_loss"},
        console_show_learning_rate=False,
        console_show_events=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    output = capsys.readouterr().out

    assert "Fit started" in output
    assert "lr=" not in output
    assert "steps_per_second" not in output
    assert "Monitor improved" not in output


def test_train_step_and_epoch_records_include_learning_rate():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[logger],
        log_every_n_steps=1,
        enable_console_logging=False,
    )

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    step_record = next(event for event in logger.events if event["event"] == "train_step")
    epoch_record = next(event for event in logger.events if event["event"] == "epoch_end")

    assert step_record["metrics"]["lr"] == 0.1
    assert epoch_record["metrics"]["lr"] == 0.1
