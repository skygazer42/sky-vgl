import pytest
import torch
from torch import nn
from vgl._artifact import ARTIFACT_FORMAT_KEY, ARTIFACT_FORMAT_VERSION_KEY

from vgl import Graph
from vgl.engine import (
    ASAM,
    AdaptiveGradientClipping,
    BootstrapBetaScheduler,
    Callback,
    ConfidencePenaltyScheduler,
    DeferredReweighting,
    FocalGammaScheduler,
    FloodingLevelScheduler,
    GeneralizedCrossEntropyScheduler,
    GradientAccumulationScheduler,
    GradientNoiseInjection,
    LabelSmoothingScheduler,
    LdamMarginScheduler,
    LogitAdjustTauScheduler,
    ModelCheckpoint,
    ExponentialMovingAverage,
    GSAM,
    GradualUnfreezing,
    LayerwiseLrDecay,
    Lookahead,
    Poly1EpsilonScheduler,
    PosWeightScheduler,
    SAM,
    SymmetricCrossEntropyBetaScheduler,
    WarmupCosineScheduler,
    WeightDecayScheduler,
)
from vgl.engine.checkpoints import CHECKPOINT_FORMAT, CHECKPOINT_FORMAT_VERSION
from vgl.train.task import Task
from vgl.train.tasks import BootstrapTask
from vgl.train.tasks import ConfidencePenaltyTask
from vgl.train.tasks import FloodingTask
from vgl.train.tasks import GeneralizedCrossEntropyTask
from vgl.train.tasks import LinkPredictionTask
from vgl.train.tasks import NodeClassificationTask
from vgl.train.tasks import Poly1CrossEntropyTask
from vgl.train.tasks import SymmetricCrossEntropyTask
from vgl.train.trainer import Trainer


class TinyNodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.x)


class TinyBinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, batch):
        return self.linear(batch.x).view(-1)


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(3, 4),
        y=torch.tensor([0, 1, 1]),
        train_mask=torch.tensor([True, True, True]),
        val_mask=torch.tensor([True, True, True]),
        test_mask=torch.tensor([True, True, True]),
    )


def test_trainer_fit_returns_structured_history_and_metrics():
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        metrics=["accuracy"],
    )
    trainer = Trainer(
        model=TinyNodeModel(),
        task=task,
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=2,
    )

    history = trainer.fit(_graph(), val_data=_graph())

    assert history["epochs"] == 2
    assert len(history["train"]) == 2
    assert len(history["val"]) == 2
    assert "loss" in history["train"][0]
    assert "accuracy" in history["val"][0]
    assert history["monitor"] == "val_loss"


class ToyBatch:
    def __init__(self, target):
        self.target = torch.tensor([target], dtype=torch.float32)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, batch):
        return self.weight.repeat(batch.target.size(0))


class PositiveToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, batch):
        return self.weight.repeat(batch.target.size(0))


class ToyTask(Task):
    def loss(self, batch, predictions, stage):
        del stage
        return ((predictions - batch.target) ** 2).mean()

    def targets(self, batch, stage):
        del stage
        return batch.target


class FineTuneBatch:
    def __init__(self, value, target):
        self.x = torch.tensor([[value]], dtype=torch.float32)
        self.target = torch.tensor([target], dtype=torch.float32)


class BinaryBatch:
    def __init__(self, values, labels):
        self.x = torch.tensor(values, dtype=torch.float32).view(-1, 1)
        self.labels = torch.tensor(labels, dtype=torch.float32)


class FineTuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_bottom = nn.Linear(1, 1, bias=False)
        self.encoder_top = nn.Linear(1, 1, bias=False)
        self.head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.encoder_bottom.weight.fill_(1.0)
            self.encoder_top.weight.fill_(1.0)
            self.head.weight.fill_(0.0)

    def forward(self, batch):
        x = self.encoder_bottom(batch.x)
        x = self.encoder_top(x)
        x = self.head(x)
        return x.view(-1)


class SaveAndStop(Callback):
    def __init__(self, path, stop_epoch):
        self.path = path
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del train_summary, val_summary
        if epoch == self.stop_epoch:
            trainer.save_training_checkpoint(self.path, history=history, metadata={"phase": "resume"})
            raise RuntimeError("pause")


class RestoreAwareCounterCallback(Callback):
    def __init__(self):
        self.counter = 0
        self.fit_start_seen_counter = None

    def state_dict(self):
        return {"counter": self.counter}

    def load_state_dict(self, state):
        self.counter = int(state["counter"])

    def on_fit_start(self, trainer, history):
        del trainer, history
        self.fit_start_seen_counter = self.counter

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, epoch, train_summary, val_summary, history
        self.counter += 1


def test_evaluate_and_test_do_not_step_optimizer():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )
    before = trainer.model.weight.detach().clone()

    trainer.evaluate([ToyBatch(1.0)], stage="val")
    trainer.test([ToyBatch(1.0)])

    after = trainer.model.weight.detach().clone()

    assert torch.equal(before, after)


def test_trainer_evaluate_avoids_tensor_item_for_loss_aggregation(monkeypatch):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    def fail_item(self):
        raise AssertionError("trainer loss aggregation should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    summary = trainer.evaluate([ToyBatch(2.0)], stage="val")

    assert summary["loss"] == pytest.approx(4.0)


def test_trainer_restores_best_state_and_saves_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=3,
        monitor="val_loss",
        save_best_path=checkpoint,
    )

    history = trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])
    saved_checkpoint = Trainer.load_checkpoint(checkpoint)

    assert history["best_epoch"] == 2
    assert torch.equal(trainer.model.weight.detach(), torch.tensor([0.0]))
    assert saved_checkpoint["format"] == "vgl.trainer_checkpoint"
    assert saved_checkpoint["format_version"] == 1
    assert saved_checkpoint["metadata"] == {
        "best_epoch": 2,
        "best_metric": 1.0,
        "monitor": "val_loss",
    }
    assert torch.equal(saved_checkpoint["model_state_dict"]["weight"], torch.tensor([0.0]))


def test_trainer_load_checkpoint_uses_weights_only_by_default(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"stub")
    captured = {}

    def fake_load(path, *, map_location=None, weights_only=False):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return {"weight": torch.tensor([1.0])}

    monkeypatch.setattr(torch, "load", fake_load)

    checkpoint_payload = Trainer.load_checkpoint(checkpoint)

    assert checkpoint_payload["format"] == "legacy.state_dict"
    assert torch.equal(checkpoint_payload["model_state_dict"]["weight"], torch.tensor([1.0]))
    assert captured == {
        "path": checkpoint,
        "map_location": None,
        "weights_only": True,
    }


def test_trainer_save_and_load_checkpoint_round_trip(tmp_path):
    checkpoint = tmp_path / "manual.pt"

    Trainer.save_checkpoint(
        checkpoint,
        {"weight": torch.tensor([3.0])},
        metadata={"epoch": 4, "note": "manual"},
    )
    payload = Trainer.load_checkpoint(checkpoint)

    assert payload == {
        "format": "vgl.trainer_checkpoint",
        "format_version": 1,
        "model_state_dict": {"weight": torch.tensor([3.0])},
        "metadata": {"epoch": 4, "note": "manual"},
    }


def test_trainer_load_checkpoint_normalizes_legacy_state_dict_files(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"weight": torch.tensor([5.0])}, checkpoint)

    payload = Trainer.load_checkpoint(checkpoint)

    assert payload == {
        "format": "legacy.state_dict",
        "format_version": 0,
        "model_state_dict": {"weight": torch.tensor([5.0])},
        "metadata": {},
    }


def test_trainer_restore_checkpoint_loads_structured_checkpoint_into_model(tmp_path):
    checkpoint = tmp_path / "structured.pt"
    model = ToyModel()

    Trainer.save_checkpoint(
        checkpoint,
        {"weight": torch.tensor([7.0])},
        metadata={"epoch": 3},
    )
    payload = Trainer.restore_checkpoint(model, checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([7.0]))
    assert payload == {
        "format": "vgl.trainer_checkpoint",
        "format_version": 1,
        "model_state_dict": {"weight": torch.tensor([7.0])},
        "metadata": {"epoch": 3},
    }


def test_trainer_restore_checkpoint_loads_legacy_state_dict_into_model(tmp_path):
    checkpoint = tmp_path / "legacy-restore.pt"
    model = ToyModel()
    torch.save({"weight": torch.tensor([9.0])}, checkpoint)

    payload = Trainer.restore_checkpoint(model, checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([9.0]))
    assert payload == {
        "format": "legacy.state_dict",
        "format_version": 0,
        "model_state_dict": {"weight": torch.tensor([9.0])},
        "metadata": {},
    }


def test_trainer_rejects_val_monitor_without_val_data():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        monitor="val_loss",
    )

    with pytest.raises(ValueError, match="val_data"):
        trainer.fit([ToyBatch(2.0)])


def test_trainer_can_resume_full_training_state_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "resume.pt"
    paused_callback = ExponentialMovingAverage(decay=0.5, apply_on_fit_end=True)
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)])

    resumed_callback = ExponentialMovingAverage(decay=0.5, apply_on_fit_end=True)
    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(2.0)])

    uninterrupted_callback = ExponentialMovingAverage(decay=0.5, apply_on_fit_end=True)
    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(2.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert len(resumed_history["train"]) == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.num_updates == uninterrupted_callback.num_updates == 4
    assert torch.allclose(resumed_trainer.model.weight.detach(), uninterrupted_trainer.model.weight.detach())
    assert resumed_trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        uninterrupted_trainer.optimizer.param_groups[0]["lr"]
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_resume_preserves_profiler_totals_across_checkpoint(tmp_path):
    checkpoint = tmp_path / "resume-profiler.pt"
    paused_callback = SaveAndStop(checkpoint, stop_epoch=2)
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        profiler="simple",
        callbacks=[paused_callback],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)])

    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        profiler="simple",
    )
    resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(2.0)])

    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        profiler="simple",
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(2.0)])

    assert resumed_history["profile"]["train_step_count"] == uninterrupted_history["profile"]["train_step_count"] == 4
    assert resumed_history["profile"]["train_stage_seconds_total"] >= 0.0
    assert resumed_history["fit_elapsed_seconds"] is not None


def test_trainer_restores_callback_state_before_on_fit_start_on_resume(tmp_path):
    checkpoint = tmp_path / "callback-order-resume.pt"
    paused_callback = RestoreAwareCounterCallback()
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=3,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=1)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)])

    resumed_callback = RestoreAwareCounterCallback()
    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=3,
        callbacks=[resumed_callback],
    )
    resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_trainer.fit([ToyBatch(2.0)])

    assert resumed_callback.fit_start_seen_counter == 1


def test_trainer_restore_training_checkpoint_rejects_negative_global_step_before_mutating_model(tmp_path):
    checkpoint = tmp_path / "bad-global-step.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"global_step": -1},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="global_step"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_integer_global_step_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-global-step-type.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"global_step": "bad"},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="global_step"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_mapping_best_state_dict(tmp_path):
    checkpoint = tmp_path / "bad-best-state.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"best_state_dict": ["bad-state"]},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="best_state_dict"):
        trainer.restore_training_checkpoint(checkpoint)


def test_trainer_restore_training_checkpoint_rejects_callback_target_mismatch_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-callback-target.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "callback_states": [
                {
                    "callback": "vgl.engine.callbacks.ExponentialMovingAverage",
                    "index": 0,
                    "state": {"shadow_state": None, "num_updates": 1},
                }
            ],
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[],
    )

    with pytest.raises(ValueError, match="checkpoint callback state does not match configured callbacks"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_mapping_callback_state_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-callback-shape.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "callback_states": ["bad"],
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[],
    )

    with pytest.raises(ValueError, match="callback_states"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_mapping_callback_payload_state_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-callback-state-payload.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "callback_states": [
                {
                    "callback": "vgl.engine.callbacks.ExponentialMovingAverage",
                    "index": 0,
                    "state": ["bad"],
                }
            ],
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[],
    )

    with pytest.raises(ValueError, match="callback_states"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_string_callback_name_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-callback-name.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "callback_states": [
                {
                    "callback": 123,
                    "index": 0,
                    "state": {},
                }
            ],
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[],
    )

    with pytest.raises(ValueError, match="callback_states"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_negative_callback_index_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-callback-index.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "callback_states": [
                {
                    "callback": "vgl.engine.callbacks.ExponentialMovingAverage",
                    "index": -1,
                    "state": {},
                }
            ],
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[],
    )

    with pytest.raises(ValueError, match="callback_states"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_integer_callback_index_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-callback-index-type.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "callback_states": [
                {
                    "callback": "vgl.engine.callbacks.ExponentialMovingAverage",
                    "index": "bad",
                    "state": {},
                }
            ],
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[],
    )

    with pytest.raises(ValueError, match="callback_states"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_missing_lr_scheduler_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-lr-scheduler.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "lr_scheduler_state_dict": {"last_epoch": 3},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="lr_scheduler_state_dict"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_missing_grad_scaler_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-grad-scaler.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "grad_scaler_state_dict": {"scale": 1024.0},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="grad_scaler_state_dict"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_best_epoch_past_resumed_history(tmp_path):
    checkpoint = tmp_path / "bad-best-epoch.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
            },
            "trainer_state": {"best_epoch": 3, "best_metric": 1.0},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="best_epoch"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_active_monitor_mismatch_between_history_and_trainer_state(
    tmp_path,
):
    checkpoint = tmp_path / "bad-active-monitor.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
            },
            "trainer_state": {"active_monitor": "val_loss"},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="active_monitor"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_global_step_behind_completed_epochs(
    tmp_path,
):
    checkpoint = tmp_path / "bad-global-step-range.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
            },
            "trainer_state": {"global_step": 1},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="global_step"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_integer_global_step_with_history_state(
    tmp_path,
):
    checkpoint = tmp_path / "bad-global-step-history-type.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
            },
            "trainer_state": {"global_step": "bad"},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="global_step"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_best_metric_without_best_epoch(
    tmp_path,
):
    checkpoint = tmp_path / "bad-best-metric.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"best_metric": 1.0},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="best_metric"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_numeric_best_metric_with_history_state(
    tmp_path,
):
    checkpoint = tmp_path / "bad-best-metric-history-type.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "best_epoch": 2,
                "best_metric": 1.0,
            },
            "trainer_state": {
                "best_epoch": 2,
                "best_metric": "bad",
                "best_state_dict": {"weight": torch.tensor([7.0])},
            },
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="best_metric"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_zero_best_epoch(
    tmp_path,
):
    checkpoint = tmp_path / "bad-zero-best-epoch.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"best_epoch": 0, "best_metric": 1.0},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="best_epoch"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_integer_best_epoch(
    tmp_path,
):
    checkpoint = tmp_path / "bad-best-epoch-type.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"best_epoch": "bad", "best_metric": 1.0, "best_state_dict": {"weight": torch.tensor([7.0])}},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="best_epoch"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_best_epoch_without_best_metric_without_history_state(
    tmp_path,
):
    checkpoint = tmp_path / "bad-best-epoch-without-metric.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"best_epoch": 1},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="best_metric"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_best_epoch_without_best_state_dict(
    tmp_path,
):
    checkpoint = tmp_path / "bad-best-epoch-without-best-state.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"best_epoch": 1, "best_metric": 1.0},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="best_state_dict"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_best_state_dict_without_best_epoch(
    tmp_path,
):
    checkpoint = tmp_path / "bad-best-state-without-epoch.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"best_state_dict": {"weight": torch.tensor([7.0])}},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match="best_state_dict"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_best_epoch_mismatch_between_history_and_trainer_state(
    tmp_path,
):
    checkpoint = tmp_path / "bad-best-epoch-mismatch.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "best_epoch": 2,
                "best_metric": 1.0,
            },
            "trainer_state": {"best_epoch": 1, "best_metric": 1.0},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="best_epoch"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_missing_trainer_best_epoch_when_history_has_one(
    tmp_path,
):
    checkpoint = tmp_path / "bad-missing-trainer-best-epoch.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "best_epoch": 2,
                "best_metric": 1.0,
            },
            "trainer_state": {},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="best_epoch"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_best_metric_mismatch_between_history_and_trainer_state(
    tmp_path,
):
    checkpoint = tmp_path / "bad-best-metric-mismatch.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "best_epoch": 2,
                "best_metric": 2.0,
            },
            "trainer_state": {"best_epoch": 2, "best_metric": 1.0},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="best_metric"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_missing_best_metric_from_trainer_state_when_history_has_one(
    tmp_path,
):
    checkpoint = tmp_path / "missing-trainer-best-metric.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "best_epoch": 2,
                "best_metric": 1.0,
            },
            "trainer_state": {"best_epoch": 2},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="best_metric"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_missing_history_best_metric_when_trainer_has_one(
    tmp_path,
):
    checkpoint = tmp_path / "missing-history-best-metric.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "best_epoch": 2,
            },
            "trainer_state": {"best_epoch": 2, "best_metric": 1.0},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="best_metric"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_best_epoch_without_best_metric(
    tmp_path,
):
    checkpoint = tmp_path / "best-epoch-without-best-metric.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "best_epoch": 2,
            },
            "trainer_state": {"best_epoch": 2},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="best_metric"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_preflights_invalid_history_state_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-history-preflight.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
            },
            "trainer_state": {},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="train history length"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_non_mapping_history_final_train_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-history-final-train.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "final_train": ["bad"],
            },
            "trainer_state": {},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="final_train"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_val_history_length_past_completed_epochs_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-history-val-length.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "val_loss",
                "completed_epochs": 1,
                "train": [{"loss": 4.0}],
                "val": [{"loss": 3.0}, {"loss": 2.0}],
                "epoch_elapsed_seconds": [0.1],
            },
            "trainer_state": {},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="val history length"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_negative_history_fit_elapsed_seconds_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-history-fit-elapsed.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "fit_elapsed_seconds": -1.0,
            },
            "trainer_state": {},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="fit_elapsed_seconds"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_negative_history_profile_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-history-profile.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "profiler": "simple",
                "profile": {"forward_seconds_total": -1.0},
            },
            "trainer_state": {},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="profile"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_history_profile_without_simple_profiler_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-history-profile-without-profiler.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "profile": {"forward_seconds_total": 1.0},
            },
            "trainer_state": {},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="profile"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_unsupported_history_profiler_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-history-profiler.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "history_state": {
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 4.0}, {"loss": 4.0}],
                "epoch_elapsed_seconds": [0.1, 0.1],
                "profiler": "advanced",
            },
            "trainer_state": {},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="profiler"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_negative_fit_profile_total_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-fit-profile-total.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"fit_profile": {"forward_seconds_total": -1.0}},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="fit_profile"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_restore_training_checkpoint_rejects_negative_fit_profile_count_before_mutating_model(
    tmp_path,
):
    checkpoint = tmp_path / "bad-fit-profile-count.pt"
    model = ToyModel()
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([9.0])},
            "metadata": {},
            "trainer_state": {"fit_profile": {"train_step_count": -1}},
        },
        checkpoint,
    )
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
    )

    with pytest.raises(ValueError, match="fit_profile"):
        trainer.restore_training_checkpoint(checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([0.0]))


def test_trainer_can_resume_lookahead_callback_state_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "lookahead-resume.pt"
    paused_callback = Lookahead(sync_period=2, slow_step_size=0.5)
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)])

    resumed_callback = Lookahead(sync_period=2, slow_step_size=0.5)
    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(2.0)])

    uninterrupted_callback = Lookahead(sync_period=2, slow_step_size=0.5)
    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(2.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.step_count == uninterrupted_callback.step_count == 4
    assert torch.allclose(resumed_callback.slow_state["weight"], uninterrupted_callback.slow_state["weight"])
    assert torch.allclose(resumed_trainer.model.weight.detach(), uninterrupted_trainer.model.weight.detach())
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_sam_optimizer_state_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "sam-resume.pt"

    def optimizer_factory(params, lr):
        return SAM(params, torch.optim.Adam, lr=lr, rho=0.05)

    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=optimizer_factory,
        lr=0.1,
        max_epochs=4,
        callbacks=[SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)])

    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=optimizer_factory,
        lr=0.1,
        max_epochs=4,
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(2.0)])

    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=optimizer_factory,
        lr=0.1,
        max_epochs=4,
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(2.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert torch.allclose(resumed_trainer.model.weight.detach(), uninterrupted_trainer.model.weight.detach())
    assert resumed_trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        uninterrupted_trainer.optimizer.param_groups[0]["lr"]
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_gsam_optimizer_state_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "gsam-resume.pt"

    def optimizer_factory(params, lr):
        return GSAM(params, torch.optim.Adam, lr=lr, rho=0.05, alpha=0.2)

    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=optimizer_factory,
        lr=0.1,
        max_epochs=4,
        callbacks=[SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)])

    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=optimizer_factory,
        lr=0.1,
        max_epochs=4,
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(2.0)])

    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=optimizer_factory,
        lr=0.1,
        max_epochs=4,
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(2.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert torch.allclose(resumed_trainer.model.weight.detach(), uninterrupted_trainer.model.weight.detach())
    assert resumed_trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        uninterrupted_trainer.optimizer.param_groups[0]["lr"]
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_asam_reuses_sam_resume_path_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "asam-resume.pt"

    def optimizer_factory(params, lr):
        return ASAM(params, torch.optim.Adam, lr=lr, rho=0.05)

    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=optimizer_factory,
        lr=0.1,
        max_epochs=4,
        callbacks=[SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)])

    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=optimizer_factory,
        lr=0.1,
        max_epochs=4,
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(2.0)])

    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=optimizer_factory,
        lr=0.1,
        max_epochs=4,
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(2.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_trainer.optimizer.param_groups[0]["adaptive"] is True
    assert torch.allclose(resumed_trainer.model.weight.detach(), uninterrupted_trainer.model.weight.detach())
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_warmup_cosine_scheduler_state_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "warmup-cosine-resume.pt"
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=5,
        lr_scheduler=lambda optimizer: WarmupCosineScheduler(
            optimizer,
            warmup_epochs=2,
            max_epochs=5,
            min_lr_ratio=0.1,
        ),
        callbacks=[SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)])

    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=5,
        lr_scheduler=lambda optimizer: WarmupCosineScheduler(
            optimizer,
            warmup_epochs=2,
            max_epochs=5,
            min_lr_ratio=0.1,
        ),
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(2.0)])

    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=5,
        lr_scheduler=lambda optimizer: WarmupCosineScheduler(
            optimizer,
            warmup_epochs=2,
            max_epochs=5,
            min_lr_ratio=0.1,
        ),
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(2.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 5
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 5
    assert resumed_trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        uninterrupted_trainer.optimizer.param_groups[0]["lr"]
    )
    assert resumed_trainer.lr_scheduler.get_last_lr() == pytest.approx(
        uninterrupted_trainer.lr_scheduler.get_last_lr()
    )
    assert torch.allclose(resumed_trainer.model.weight.detach(), uninterrupted_trainer.model.weight.detach())
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_gradual_unfreezing_callback_state_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "gradual-unfreezing-resume.pt"
    paused_callback = GradualUnfreezing(["encoder_top", "encoder_bottom"], start_epoch=2, frequency=1)
    paused_trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([FineTuneBatch(1.0, 2.0)])

    resumed_callback = GradualUnfreezing(["encoder_top", "encoder_bottom"], start_epoch=2, frequency=1)
    resumed_trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([FineTuneBatch(1.0, 2.0)])

    uninterrupted_callback = GradualUnfreezing(["encoder_top", "encoder_bottom"], start_epoch=2, frequency=1)
    uninterrupted_trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit([FineTuneBatch(1.0, 2.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.unfrozen_group_count == uninterrupted_callback.unfrozen_group_count == 2
    assert (
        resumed_trainer.model.encoder_bottom.weight.requires_grad
        == uninterrupted_trainer.model.encoder_bottom.weight.requires_grad
    )
    assert resumed_trainer.model.encoder_top.weight.requires_grad == uninterrupted_trainer.model.encoder_top.weight.requires_grad
    assert torch.allclose(resumed_trainer.model.encoder_bottom.weight.detach(), uninterrupted_trainer.model.encoder_bottom.weight.detach())
    assert torch.allclose(resumed_trainer.model.encoder_top.weight.detach(), uninterrupted_trainer.model.encoder_top.weight.detach())
    assert torch.allclose(resumed_trainer.model.head.weight.detach(), uninterrupted_trainer.model.head.weight.detach())
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_custom_optimizer_param_groups_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "layerwise-lr-resume.pt"
    param_groups = LayerwiseLrDecay(
        ["head", "encoder_top", "encoder_bottom"],
        lr_decay=0.5,
        include_rest=False,
    )
    paused_trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.Adam,
        lr=0.1,
        max_epochs=4,
        optimizer_param_groups=param_groups,
        callbacks=[SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([FineTuneBatch(1.0, 2.0)])

    resumed_trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.Adam,
        lr=0.1,
        max_epochs=4,
        optimizer_param_groups=param_groups,
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([FineTuneBatch(1.0, 2.0)])

    uninterrupted_trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.Adam,
        lr=0.1,
        max_epochs=4,
        optimizer_param_groups=param_groups,
    )
    uninterrupted_history = uninterrupted_trainer.fit([FineTuneBatch(1.0, 2.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert len(resumed_trainer.optimizer.param_groups) == len(uninterrupted_trainer.optimizer.param_groups) == 3
    assert [group["lr"] for group in resumed_trainer.optimizer.param_groups] == pytest.approx(
        [group["lr"] for group in uninterrupted_trainer.optimizer.param_groups]
    )
    assert [group["group_name"] for group in resumed_trainer.optimizer.param_groups] == [
        group["group_name"] for group in uninterrupted_trainer.optimizer.param_groups
    ]
    assert torch.allclose(resumed_trainer.model.encoder_bottom.weight.detach(), uninterrupted_trainer.model.encoder_bottom.weight.detach())
    assert torch.allclose(resumed_trainer.model.encoder_top.weight.detach(), uninterrupted_trainer.model.encoder_top.weight.detach())
    assert torch.allclose(resumed_trainer.model.head.weight.detach(), uninterrupted_trainer.model.head.weight.detach())
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_step_wise_scheduler_state_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "step-scheduler-resume.pt"
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5,
        ),
        lr_scheduler_interval="step",
        callbacks=[SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)])

    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5,
        ),
        lr_scheduler_interval="step",
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(2.0)])

    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5,
        ),
        lr_scheduler_interval="step",
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(2.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        uninterrupted_trainer.optimizer.param_groups[0]["lr"]
    )
    assert torch.allclose(resumed_trainer.model.weight.detach(), uninterrupted_trainer.model.weight.detach())
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_adaptive_gradient_clipping_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "agc-resume.pt"
    paused_callback = AdaptiveGradientClipping(clipping=0.1, eps=1e-3)
    paused_trainer = Trainer(
        model=PositiveToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(10.0)])

    resumed_callback = AdaptiveGradientClipping(clipping=0.1, eps=1e-3)
    resumed_trainer = Trainer(
        model=PositiveToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(10.0)])

    uninterrupted_callback = AdaptiveGradientClipping(clipping=0.1, eps=1e-3)
    uninterrupted_trainer = Trainer(
        model=PositiveToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(10.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert torch.allclose(resumed_trainer.model.weight.detach(), uninterrupted_trainer.model.weight.detach())
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_deferred_reweighting_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "drw-resume.pt"
    torch.manual_seed(7)
    graph = _graph()
    torch.manual_seed(13)
    paused_callback = DeferredReweighting(start_epoch=2, beta=0.9)
    paused_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        class_count=[1.0, 2.0],
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(11)
    resumed_callback = DeferredReweighting(start_epoch=2, beta=0.9)
    resumed_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        class_count=[1.0, 2.0],
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    uninterrupted_callback = DeferredReweighting(start_epoch=2, beta=0.9)
    torch.manual_seed(13)
    uninterrupted_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        class_count=[1.0, 2.0],
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.reweighting_active == uninterrupted_callback.reweighting_active is True
    assert resumed_task.class_weight is None
    assert uninterrupted_task.class_weight is None
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_label_smoothing_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "label-smoothing-resume.pt"
    torch.manual_seed(19)
    graph = _graph()
    torch.manual_seed(23)
    paused_callback = LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=2, end_epoch=4)
    paused_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        label_smoothing=0.05,
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(31)
    resumed_callback = LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=2, end_epoch=4)
    resumed_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        label_smoothing=0.05,
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(23)
    uninterrupted_callback = LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=2, end_epoch=4)
    uninterrupted_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        label_smoothing=0.05,
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(0.2)
    assert resumed_task.label_smoothing == pytest.approx(0.05)
    assert uninterrupted_task.label_smoothing == pytest.approx(0.05)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_focal_gamma_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "focal-gamma-resume.pt"
    torch.manual_seed(29)
    graph = _graph()
    torch.manual_seed(37)
    paused_callback = FocalGammaScheduler(start_value=0.5, end_value=3.0, start_epoch=2, end_epoch=4)
    paused_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        loss="focal",
        focal_gamma=1.5,
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(41)
    resumed_callback = FocalGammaScheduler(start_value=0.5, end_value=3.0, start_epoch=2, end_epoch=4)
    resumed_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        loss="focal",
        focal_gamma=1.5,
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(37)
    uninterrupted_callback = FocalGammaScheduler(start_value=0.5, end_value=3.0, start_epoch=2, end_epoch=4)
    uninterrupted_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        loss="focal",
        focal_gamma=1.5,
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(3.0)
    assert resumed_task.focal_gamma == pytest.approx(1.5)
    assert uninterrupted_task.focal_gamma == pytest.approx(1.5)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_logit_adjust_tau_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "logit-adjust-tau-resume.pt"
    torch.manual_seed(43)
    graph = _graph()
    torch.manual_seed(47)
    paused_callback = LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=2, end_epoch=4)
    paused_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        loss="logit_adjustment",
        class_count=[1.0, 2.0],
        logit_adjust_tau=0.2,
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(53)
    resumed_callback = LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=2, end_epoch=4)
    resumed_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        loss="logit_adjustment",
        class_count=[1.0, 2.0],
        logit_adjust_tau=0.2,
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(47)
    uninterrupted_callback = LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=2, end_epoch=4)
    uninterrupted_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        loss="logit_adjustment",
        class_count=[1.0, 2.0],
        logit_adjust_tau=0.2,
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(1.5)
    assert resumed_task.logit_adjust_tau == pytest.approx(0.2)
    assert uninterrupted_task.logit_adjust_tau == pytest.approx(0.2)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_ldam_margin_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "ldam-margin-resume.pt"
    torch.manual_seed(59)
    graph = _graph()
    torch.manual_seed(61)
    paused_callback = LdamMarginScheduler(start_value=0.2, end_value=0.5, start_epoch=2, end_epoch=4)
    paused_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        loss="ldam",
        class_count=[1.0, 2.0],
        ldam_max_margin=0.35,
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(67)
    resumed_callback = LdamMarginScheduler(start_value=0.2, end_value=0.5, start_epoch=2, end_epoch=4)
    resumed_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        loss="ldam",
        class_count=[1.0, 2.0],
        ldam_max_margin=0.35,
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(61)
    uninterrupted_callback = LdamMarginScheduler(start_value=0.2, end_value=0.5, start_epoch=2, end_epoch=4)
    uninterrupted_task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        loss="ldam",
        class_count=[1.0, 2.0],
        ldam_max_margin=0.35,
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(0.5)
    assert resumed_task.ldam_max_margin == pytest.approx(0.35)
    assert uninterrupted_task.ldam_max_margin == pytest.approx(0.35)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_pos_weight_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "pos-weight-resume.pt"
    batch = BinaryBatch(values=[1.0, 0.0, 2.0], labels=[1.0, 0.0, 1.0])
    torch.manual_seed(71)
    paused_callback = PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=2, end_epoch=4)
    paused_task = LinkPredictionTask(target="label", pos_weight=2.0)
    paused_trainer = Trainer(
        model=TinyBinaryModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([batch])

    torch.manual_seed(73)
    resumed_callback = PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=2, end_epoch=4)
    resumed_task = LinkPredictionTask(target="label", pos_weight=2.0)
    resumed_trainer = Trainer(
        model=TinyBinaryModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([batch])

    torch.manual_seed(71)
    uninterrupted_callback = PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=2, end_epoch=4)
    uninterrupted_task = LinkPredictionTask(target="label", pos_weight=2.0)
    uninterrupted_trainer = Trainer(
        model=TinyBinaryModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit([batch])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(4.0)
    assert torch.allclose(resumed_task.pos_weight, torch.tensor([2.0]))
    assert torch.allclose(uninterrupted_task.pos_weight, torch.tensor([2.0]))
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_bootstrap_beta_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "bootstrap-beta-resume.pt"
    torch.manual_seed(75)
    graph = _graph()
    torch.manual_seed(77)
    paused_callback = BootstrapBetaScheduler(start_value=0.2, end_value=0.8, start_epoch=2, end_epoch=4)
    paused_task = BootstrapTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        beta=0.95,
        mode="soft",
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(81)
    resumed_callback = BootstrapBetaScheduler(start_value=0.2, end_value=0.8, start_epoch=2, end_epoch=4)
    resumed_task = BootstrapTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        beta=0.95,
        mode="soft",
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(77)
    uninterrupted_callback = BootstrapBetaScheduler(start_value=0.2, end_value=0.8, start_epoch=2, end_epoch=4)
    uninterrupted_task = BootstrapTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        beta=0.95,
        mode="soft",
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(0.8)
    assert resumed_task.beta == pytest.approx(0.95)
    assert uninterrupted_task.beta == pytest.approx(0.95)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_confidence_penalty_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "confidence-penalty-resume.pt"
    torch.manual_seed(85)
    graph = _graph()
    torch.manual_seed(89)
    paused_callback = ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    paused_task = ConfidencePenaltyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        coefficient=0.1,
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(97)
    resumed_callback = ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    resumed_task = ConfidencePenaltyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        coefficient=0.1,
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(89)
    uninterrupted_callback = ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    uninterrupted_task = ConfidencePenaltyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        coefficient=0.1,
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(0.3)
    assert resumed_task.coefficient == pytest.approx(0.1)
    assert uninterrupted_task.coefficient == pytest.approx(0.1)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_flooding_level_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "flooding-level-resume.pt"
    torch.manual_seed(101)
    graph = _graph()
    torch.manual_seed(103)
    paused_callback = FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    paused_task = FloodingTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        level=0.1,
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(107)
    resumed_callback = FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    resumed_task = FloodingTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        level=0.1,
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(103)
    uninterrupted_callback = FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    uninterrupted_task = FloodingTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        level=0.1,
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(0.3)
    assert resumed_task.level == pytest.approx(0.1)
    assert uninterrupted_task.level == pytest.approx(0.1)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_generalized_cross_entropy_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "generalized-cross-entropy-resume.pt"
    torch.manual_seed(109)
    graph = _graph()
    torch.manual_seed(113)
    paused_callback = GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    paused_task = GeneralizedCrossEntropyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        q=0.7,
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(127)
    resumed_callback = GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    resumed_task = GeneralizedCrossEntropyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        q=0.7,
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(113)
    uninterrupted_callback = GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    uninterrupted_task = GeneralizedCrossEntropyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        q=0.7,
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(0.9)
    assert resumed_task.q == pytest.approx(0.7)
    assert uninterrupted_task.q == pytest.approx(0.7)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_poly1_epsilon_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "poly1-epsilon-resume.pt"
    torch.manual_seed(131)
    graph = _graph()
    torch.manual_seed(137)
    paused_callback = Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    paused_task = Poly1CrossEntropyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        epsilon=0.7,
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(139)
    resumed_callback = Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    resumed_task = Poly1CrossEntropyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        epsilon=0.7,
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(137)
    uninterrupted_callback = Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    uninterrupted_task = Poly1CrossEntropyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        epsilon=0.7,
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(0.9)
    assert resumed_task.epsilon == pytest.approx(0.7)
    assert uninterrupted_task.epsilon == pytest.approx(0.7)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_symmetric_cross_entropy_beta_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "symmetric-cross-entropy-beta-resume.pt"
    torch.manual_seed(131)
    graph = _graph()
    torch.manual_seed(137)
    paused_callback = SymmetricCrossEntropyBetaScheduler(
        start_value=0.3,
        end_value=0.9,
        start_epoch=2,
        end_epoch=4,
    )
    paused_task = SymmetricCrossEntropyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        alpha=1.0,
        beta=0.7,
    )
    paused_trainer = Trainer(
        model=TinyNodeModel(),
        task=paused_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit(graph)

    torch.manual_seed(139)
    resumed_callback = SymmetricCrossEntropyBetaScheduler(
        start_value=0.3,
        end_value=0.9,
        start_epoch=2,
        end_epoch=4,
    )
    resumed_task = SymmetricCrossEntropyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        alpha=1.0,
        beta=0.7,
    )
    resumed_trainer = Trainer(
        model=TinyNodeModel(),
        task=resumed_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit(graph)

    torch.manual_seed(137)
    uninterrupted_callback = SymmetricCrossEntropyBetaScheduler(
        start_value=0.3,
        end_value=0.9,
        start_epoch=2,
        end_epoch=4,
    )
    uninterrupted_task = SymmetricCrossEntropyTask(
        NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
        ),
        alpha=1.0,
        beta=0.7,
    )
    uninterrupted_trainer = Trainer(
        model=TinyNodeModel(),
        task=uninterrupted_task,
        optimizer=torch.optim.SGD,
        lr=0.05,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(graph)

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_value == uninterrupted_callback.current_value == pytest.approx(0.9)
    assert resumed_task.beta == pytest.approx(0.7)
    assert uninterrupted_task.beta == pytest.approx(0.7)
    assert torch.allclose(
        resumed_trainer.model.linear.weight.detach(),
        uninterrupted_trainer.model.linear.weight.detach(),
    )
    assert torch.allclose(
        resumed_trainer.model.linear.bias.detach(),
        uninterrupted_trainer.model.linear.bias.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_weight_decay_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "weight-decay-resume.pt"
    torch.manual_seed(79)
    paused_callback = WeightDecayScheduler(start_factor=0.0, end_factor=1.5, start_epoch=2, end_epoch=4)
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=lambda params, lr: torch.optim.SGD(params, lr=lr, weight_decay=0.2),
        lr=0.1,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(1.0)])

    torch.manual_seed(83)
    resumed_callback = WeightDecayScheduler(start_factor=0.0, end_factor=1.5, start_epoch=2, end_epoch=4)
    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=lambda params, lr: torch.optim.SGD(params, lr=lr, weight_decay=0.2),
        lr=0.1,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(1.0)])

    torch.manual_seed(79)
    uninterrupted_callback = WeightDecayScheduler(start_factor=0.0, end_factor=1.5, start_epoch=2, end_epoch=4)
    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=lambda params, lr: torch.optim.SGD(params, lr=lr, weight_decay=0.2),
        lr=0.1,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(1.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.current_factor == uninterrupted_callback.current_factor == pytest.approx(1.5)
    assert resumed_trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    assert uninterrupted_trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    assert torch.allclose(
        resumed_trainer.model.weight.detach(),
        uninterrupted_trainer.model.weight.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_gradient_noise_injection_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "gradient-noise-resume.pt"
    torch.manual_seed(89)
    paused_callback = GradientNoiseInjection(std=0.05, decay_exponent=0.5, seed=123)
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=4,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(1.0)])

    torch.manual_seed(97)
    resumed_callback = GradientNoiseInjection(std=0.05, decay_exponent=0.5, seed=123)
    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=4,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(1.0)])

    torch.manual_seed(89)
    uninterrupted_callback = GradientNoiseInjection(std=0.05, decay_exponent=0.5, seed=123)
    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=4,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(1.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.step_count == uninterrupted_callback.step_count == 4
    assert torch.allclose(
        resumed_trainer.model.weight.detach(),
        uninterrupted_trainer.model.weight.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_gradient_accumulation_scheduler_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "gradient-accumulation-resume.pt"
    paused_callback = GradientAccumulationScheduler({2: 2, 3: 3})
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=4,
        accumulate_grad_batches=1,
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(1.0), ToyBatch(2.0), ToyBatch(3.0), ToyBatch(4.0)])

    resumed_callback = GradientAccumulationScheduler({2: 2, 3: 3})
    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=4,
        accumulate_grad_batches=1,
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(1.0), ToyBatch(2.0), ToyBatch(3.0), ToyBatch(4.0)])

    uninterrupted_callback = GradientAccumulationScheduler({2: 2, 3: 3})
    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=4,
        accumulate_grad_batches=1,
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit(
        [ToyBatch(1.0), ToyBatch(2.0), ToyBatch(3.0), ToyBatch(4.0)]
    )

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 10
    assert resumed_callback.current_accumulate_grad_batches == uninterrupted_callback.current_accumulate_grad_batches == 3
    assert resumed_trainer.accumulate_grad_batches == uninterrupted_trainer.accumulate_grad_batches == 1
    assert torch.allclose(
        resumed_trainer.model.weight.detach(),
        uninterrupted_trainer.model.weight.detach(),
    )
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]


def test_trainer_can_resume_model_checkpoint_callback_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "model-checkpoint-resume.pt"
    checkpoint_dir = tmp_path / "managed-checkpoints"
    paused_callback = ModelCheckpoint(
        checkpoint_dir,
        filename="epoch{epoch}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
    )
    paused_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        monitor="val_loss",
        callbacks=[paused_callback, SaveAndStop(checkpoint, stop_epoch=2)],
    )

    with pytest.raises(RuntimeError, match="pause"):
        paused_trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    resumed_callback = ModelCheckpoint(
        checkpoint_dir,
        filename="epoch{epoch}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
    )
    resumed_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        monitor="val_loss",
        callbacks=[resumed_callback],
    )
    resumed_payload = resumed_trainer.restore_training_checkpoint(checkpoint)
    resumed_history = resumed_trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    uninterrupted_callback = ModelCheckpoint(
        tmp_path / "uninterrupted-checkpoints",
        filename="epoch{epoch}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
    )
    uninterrupted_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=4,
        monitor="val_loss",
        callbacks=[uninterrupted_callback],
    )
    uninterrupted_history = uninterrupted_trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    assert resumed_payload["metadata"] == {"phase": "resume"}
    assert resumed_history["completed_epochs"] == 4
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 4
    assert resumed_callback.best_model_score == pytest.approx(uninterrupted_callback.best_model_score)
    assert resumed_callback.kth_best_model_score == pytest.approx(uninterrupted_callback.kth_best_model_score)
    assert resumed_callback.best_model_path is not None
    assert resumed_callback.best_model_path.endswith("epoch2.ckpt")
    assert resumed_callback.last_model_path is not None
    assert resumed_callback.last_model_path.endswith("last.ckpt")
    assert (checkpoint_dir / "last.ckpt").exists()
    assert resumed_history["best_epoch"] == uninterrupted_history["best_epoch"]
