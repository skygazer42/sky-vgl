import json

import torch
from torch import nn

from vgl.engine import JSONLinesLogger, Logger, ModelCheckpoint, Trainer
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


class ToyTask(Task):
    def loss(self, batch, predictions, stage):
        del stage
        return ((predictions - batch.target) ** 2).mean()

    def targets(self, batch, stage):
        del stage
        return batch.target


class RecordingLogger(Logger):
    def __init__(self):
        self.records = []

    def on_fit_start(self, run_info):
        self.records.append(dict(run_info))

    def on_stage_start(self, stage_record):
        self.records.append(dict(stage_record))

    def on_train_step(self, log_record):
        self.records.append(dict(log_record))

    def on_epoch_end(self, epoch_record):
        self.records.append(dict(epoch_record))

    def on_evaluate_end(self, stage_record):
        self.records.append(dict(stage_record))

    def on_fit_end(self, final_record):
        self.records.append(dict(final_record))

    def on_exception(self, exception_record):
        self.records.append(dict(exception_record))

    def on_event(self, record):
        self.records.append(dict(record))


def test_fast_dev_run_limits_stages_to_single_batch_and_single_epoch(tmp_path):
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=5,
        loggers=[logger],
        enable_console_logging=False,
        fast_dev_run=True,
        default_root_dir=tmp_path,
        save_best_path="checkpoints/best.pt",
    )

    history = trainer.fit([ToyBatch(1.0) for _ in range(5)], val_data=[ToyBatch(1.0) for _ in range(4)])

    stage_starts = [record for record in logger.records if record["event"] == "stage_start"]

    assert history["completed_epochs"] == 1
    assert history["fast_dev_run"] is True
    assert [(record["stage"], record["total_batches"]) for record in stage_starts] == [
        ("train", 1),
        ("val", 1),
    ]
    assert not (tmp_path / "checkpoints" / "best.pt").exists()


def test_fast_dev_run_disables_model_checkpoint_callback(tmp_path):
    logger = RecordingLogger()
    callback = ModelCheckpoint("checkpoints", save_top_k=1, save_last=True)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=3,
        callbacks=[callback],
        loggers=[logger],
        enable_console_logging=False,
        default_root_dir=tmp_path,
        fast_dev_run=True,
    )

    trainer.fit([ToyBatch(1.0) for _ in range(3)], val_data=[ToyBatch(1.0)])

    checkpoint_events = [record for record in logger.records if record["event"] == "checkpoint_saved"]

    assert checkpoint_events == []
    assert not (tmp_path / "checkpoints").exists()
    assert callback.last_model_path is None
    assert callback.best_model_path is None


def test_limit_batch_parameters_trim_train_val_and_test_batches():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[logger],
        enable_console_logging=False,
        limit_train_batches=0.4,
        limit_val_batches=0.25,
        limit_test_batches=2,
    )

    trainer.fit([ToyBatch(1.0) for _ in range(5)], val_data=[ToyBatch(1.0) for _ in range(4)])
    trainer.test([ToyBatch(1.0) for _ in range(5)])

    stage_starts = [record for record in logger.records if record["event"] == "stage_start"]

    assert [(record["stage"], record["total_batches"]) for record in stage_starts] == [
        ("train", 2),
        ("val", 1),
        ("test", 2),
    ]


def test_num_sanity_val_steps_runs_validation_before_training():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[logger],
        enable_console_logging=False,
        num_sanity_val_steps=2,
    )

    history = trainer.fit([ToyBatch(1.0) for _ in range(3)], val_data=[ToyBatch(1.0) for _ in range(4)])

    stage_starts = [record for record in logger.records if record["event"] == "stage_start"]

    assert history["sanity_check_passed"] is True
    assert stage_starts[0]["stage"] == "sanity_val"
    assert stage_starts[0]["total_batches"] == 2
    assert stage_starts[1]["stage"] == "train"


def test_default_root_dir_resolves_relative_paths_and_records_run_metadata(tmp_path):
    logger_path = "logs/train.jsonl"
    checkpoint_path = "checkpoints/best.pt"
    root_dir = tmp_path / "artifacts"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[JSONLinesLogger(logger_path, flush=True)],
        enable_console_logging=False,
        default_root_dir=root_dir,
        run_name="smoke-run",
        save_best_path=checkpoint_path,
    )

    history = trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    log_path = root_dir / "logs" / "train.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines()]

    assert history["run_name"] == "smoke-run"
    assert history["root_dir"] == str(root_dir)
    assert (root_dir / "checkpoints" / "best.pt").exists()
    assert records[0]["run_name"] == "smoke-run"
    assert records[0]["root_dir"] == str(root_dir)


def test_simple_profiler_adds_profile_data_to_history_and_fit_records():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[logger],
        enable_console_logging=False,
        profiler="simple",
    )

    history = trainer.fit([ToyBatch(1.0), ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    fit_start = logger.records[0]
    fit_end = next(record for record in logger.records if record["event"] == "fit_end")
    epoch_record = next(record for record in logger.records if record["event"] == "epoch_end")

    assert fit_start["profiler"] == "simple"
    assert history["profile"]["forward_seconds_total"] >= 0.0
    assert history["profile"]["optimizer_step_seconds_total"] >= 0.0
    assert history["profile"]["train_stage_seconds_total"] >= 0.0
    assert fit_end["profile"]["train_stage_seconds_total"] >= 0.0
    assert epoch_record["profile"]["train_stage_seconds_total"] >= 0.0


def test_simple_profiler_exposes_stable_profile_schema():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        enable_console_logging=False,
        profiler="simple",
    )

    history = trainer.fit([ToyBatch(1.0), ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    assert tuple(history["profile"]) == (
        "batch_materialization_seconds_total",
        "forward_seconds_total",
        "backward_seconds_total",
        "optimizer_step_seconds_total",
        "train_step_seconds_total",
        "train_stage_seconds_total",
        "val_stage_seconds_total",
        "test_stage_seconds_total",
        "sanity_val_stage_seconds_total",
        "train_step_count",
        "train_step_seconds_avg",
    )


def test_val_check_interval_runs_mid_epoch_validation():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[logger],
        enable_console_logging=False,
        val_check_interval=0.5,
    )

    trainer.fit([ToyBatch(1.0) for _ in range(4)], val_data=[ToyBatch(1.0) for _ in range(2)])

    evaluate_end_records = [record for record in logger.records if record["event"] == "evaluate_end"]
    epoch_end = next(record for record in logger.records if record["event"] == "epoch_end")

    assert len(evaluate_end_records) == 1
    assert evaluate_end_records[0]["stage"] == "val"
    assert "val_loss" in epoch_end["metrics"]


def test_integer_val_check_interval_triggers_mid_epoch_validation():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[logger],
        enable_console_logging=False,
        val_check_interval=1,
    )

    trainer.fit([ToyBatch(1.0) for _ in range(3)], val_data=[ToyBatch(1.0)])

    evaluate_end_records = [record for record in logger.records if record["event"] == "evaluate_end"]

    assert len(evaluate_end_records) == 2
    assert all(record["stage"] == "val" for record in evaluate_end_records)
