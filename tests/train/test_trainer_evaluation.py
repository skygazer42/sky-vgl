import pytest
import torch
from torch import nn

from vgl import Graph
from vgl.engine import (
    ASAM,
    AdaptiveGradientClipping,
    Callback,
    DeferredReweighting,
    ExponentialMovingAverage,
    GSAM,
    GradualUnfreezing,
    LayerwiseLrDecay,
    Lookahead,
    SAM,
    WarmupCosineScheduler,
)
from vgl.train.task import Task
from vgl.train.tasks import NodeClassificationTask
from vgl.train.trainer import Trainer


class TinyNodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.x)


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
