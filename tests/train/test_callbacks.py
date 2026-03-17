import pytest
import torch
from torch import nn

from vgl.engine import (
    AdaptiveGradientClipping,
    Callback,
    EarlyStopping,
    ExponentialMovingAverage,
    GradientCentralization,
    GradualUnfreezing,
    HistoryLogger,
    Lookahead,
    StopTraining,
    StochasticWeightAveraging,
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


class FreezeStateRecorder(Callback):
    def __init__(self):
        self.states = []

    def _record(self, label, trainer):
        self.states.append(
            (
                label,
                {
                    "encoder_bottom.weight": trainer.model.encoder_bottom.weight.requires_grad,
                    "encoder_top.weight": trainer.model.encoder_top.weight.requires_grad,
                    "head.weight": trainer.model.head.weight.requires_grad,
                },
            )
        )

    def on_fit_start(self, trainer, history):
        del history
        self._record("fit_start", trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del train_summary, val_summary, history
        self._record(f"epoch_end_{epoch}", trainer)


class RecordingCallback(Callback):
    def __init__(self):
        self.events = []

    def on_fit_start(self, trainer, history):
        del trainer
        self.events.append(("fit_start", history["monitor"], history["completed_epochs"]))

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary
        self.events.append(("epoch_end", epoch, history["completed_epochs"], val_summary))

    def on_fit_end(self, trainer, history):
        del trainer
        self.events.append(
            ("fit_end", history["completed_epochs"], history["stopped_early"], history["stop_reason"])
        )


class StopAfterFirstEpoch(Callback):
    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary, history
        if epoch == 1:
            raise StopTraining("requested stop")


class StepRecordingCallback(Callback):
    def __init__(self):
        self.steps = []

    def on_before_optimizer_step(self, trainer, step):
        del trainer
        self.steps.append(("before", step))

    def on_after_optimizer_step(self, trainer, step):
        del trainer
        self.steps.append(("after", step))


class GradientHolderModel(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.weight = nn.Parameter(parameter)


class DummyTrainer:
    def __init__(self, model):
        self.model = model


def test_trainer_callbacks_observe_fit_lifecycle():
    callback = RecordingCallback()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=2,
        callbacks=[callback],
    )

    history = trainer.fit([ToyBatch(1.0)])

    assert history["completed_epochs"] == 2
    assert history["stopped_early"] is False
    assert history["stop_reason"] is None
    assert callback.events == [
        ("fit_start", "train_loss", 0),
        ("epoch_end", 1, 1, None),
        ("epoch_end", 2, 2, None),
        ("fit_end", 2, False, None),
    ]


def test_trainer_callbacks_can_request_early_stop():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=5,
        callbacks=[StopAfterFirstEpoch()],
    )

    history = trainer.fit([ToyBatch(1.0)])

    assert history["epochs"] == 5
    assert history["completed_epochs"] == 1
    assert history["stopped_early"] is True
    assert history["stop_reason"] == "requested stop"
    assert history["best_epoch"] == 1


def test_early_stopping_callback_stops_on_non_improving_monitor(monkeypatch):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=5,
        monitor="val_loss",
        callbacks=[EarlyStopping(patience=0)],
    )
    val_losses = iter([3.0, 2.0, 2.1, 2.2, 2.3])

    def fake_run_epoch(data, stage, training):
        del data, training
        if stage == "train":
            return {"loss": 1.0}
        return {"loss": next(val_losses)}

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)

    history = trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    assert history["completed_epochs"] == 3
    assert history["best_epoch"] == 2
    assert history["best_metric"] == 2.0
    assert history["stopped_early"] is True
    assert history["stop_reason"] == "Early stopping on val_loss"


def test_history_logger_records_epoch_summaries(monkeypatch):
    emitted = []
    logger = HistoryLogger(sink=emitted.append)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=2,
        monitor="val_loss",
        callbacks=[logger],
    )
    train_losses = iter([1.0, 0.5])
    val_losses = iter([2.0, 1.5])

    def fake_run_epoch(data, stage, training):
        del data, training
        if stage == "train":
            return {"loss": next(train_losses)}
        return {"loss": next(val_losses)}

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)

    history = trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    assert history["completed_epochs"] == 2
    assert logger.records == [
        {"epoch": 1, "train": {"loss": 1.0}, "val": {"loss": 2.0}, "best_epoch": 1, "best_metric": 2.0},
        {"epoch": 2, "train": {"loss": 0.5}, "val": {"loss": 1.5}, "best_epoch": 2, "best_metric": 1.5},
    ]
    assert emitted == logger.records


def test_trainer_calls_after_optimizer_step_once_per_optimizer_step():
    callback = StepRecordingCallback()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        accumulate_grad_batches=2,
        callbacks=[callback],
    )

    history = trainer.fit([ToyBatch(1.0), ToyBatch(3.0), ToyBatch(5.0)])

    assert history["completed_epochs"] == 1
    assert trainer.global_step == 2
    assert callback.steps == [("before", 1), ("after", 1), ("before", 2), ("after", 2)]


def test_exponential_moving_average_can_apply_shadow_weights_at_fit_end():
    callback = ExponentialMovingAverage(decay=0.5, apply_on_fit_end=True)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=2,
        callbacks=[callback],
    )

    history = trainer.fit([ToyBatch(2.0)])

    assert history["completed_epochs"] == 2
    assert callback.num_updates == 2
    assert torch.allclose(callback.shadow_state["weight"], torch.tensor([1.0]))
    assert torch.allclose(trainer.model.weight.detach(), torch.tensor([1.0]))


def test_adaptive_gradient_clipping_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="clipping"):
        AdaptiveGradientClipping(clipping=0.0)

    with pytest.raises(ValueError, match="eps"):
        AdaptiveGradientClipping(eps=0.0)


def test_gradient_centralization_rejects_invalid_configuration():
    with pytest.raises(TypeError, match="conv_only"):
        GradientCentralization(conv_only="yes")


def test_gradient_centralization_centralizes_matrix_gradients():
    callback = GradientCentralization()
    model = GradientHolderModel(torch.zeros(2, 2))
    model.weight.grad = torch.tensor([[1.0, 3.0], [2.0, 6.0]])

    callback.on_before_optimizer_step(DummyTrainer(model), step=1)

    assert torch.allclose(model.weight.grad, torch.tensor([[-1.0, 1.0], [-2.0, 2.0]]))


def test_gradient_centralization_conv_only_skips_linear_weights():
    callback = GradientCentralization(conv_only=True)
    model = GradientHolderModel(torch.zeros(2, 2))
    original_grad = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
    model.weight.grad = original_grad.clone()

    callback.on_before_optimizer_step(DummyTrainer(model), step=1)

    assert torch.allclose(model.weight.grad, original_grad)


def test_gradient_centralization_conv_only_centralizes_conv_weights():
    callback = GradientCentralization(conv_only=True)
    model = GradientHolderModel(torch.zeros(1, 1, 2, 2))
    model.weight.grad = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])

    callback.on_before_optimizer_step(DummyTrainer(model), step=1)

    assert torch.allclose(model.weight.grad, torch.tensor([[[[-3.0, -1.0], [1.0, 3.0]]]]))


def test_adaptive_gradient_clipping_limits_parameter_update():
    callback = AdaptiveGradientClipping(clipping=0.1, eps=1e-3)
    trainer = Trainer(
        model=PositiveToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[callback],
    )

    history = trainer.fit([ToyBatch(10.0)])

    assert history["completed_epochs"] == 1
    assert torch.allclose(trainer.model.weight.detach(), torch.tensor([1.1]))


def test_gradual_unfreezing_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="module_name_groups"):
        GradualUnfreezing([])

    with pytest.raises(ValueError, match="start_epoch"):
        GradualUnfreezing(["encoder"], start_epoch=0)

    with pytest.raises(ValueError, match="frequency"):
        GradualUnfreezing(["encoder"], frequency=0)


def test_gradual_unfreezing_freezes_modules_then_unfreezes_on_schedule():
    callback = GradualUnfreezing(["encoder_top", "encoder_bottom"], start_epoch=2, frequency=1)
    recorder = FreezeStateRecorder()
    trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=3,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([FineTuneBatch(1.0, 2.0)])

    assert history["completed_epochs"] == 3
    assert callback.unfrozen_group_count == 2
    assert recorder.states == [
        (
            "fit_start",
            {
                "encoder_bottom.weight": False,
                "encoder_top.weight": False,
                "head.weight": True,
            },
        ),
        (
            "epoch_end_1",
            {
                "encoder_bottom.weight": False,
                "encoder_top.weight": True,
                "head.weight": True,
            },
        ),
        (
            "epoch_end_2",
            {
                "encoder_bottom.weight": True,
                "encoder_top.weight": True,
                "head.weight": True,
            },
        ),
        (
            "epoch_end_3",
            {
                "encoder_bottom.weight": True,
                "encoder_top.weight": True,
                "head.weight": True,
            },
        ),
    ]
    assert trainer.model.encoder_top.weight.detach().item() != pytest.approx(1.0)
    assert trainer.model.encoder_bottom.weight.detach().item() != pytest.approx(1.0)


def test_gradual_unfreezing_state_dict_round_trip_preserves_progress():
    callback = GradualUnfreezing(["encoder_top", "encoder_bottom"], start_epoch=2, frequency=1)
    trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=2,
        callbacks=[callback],
    )

    trainer.fit([FineTuneBatch(1.0, 2.0)])

    restored = GradualUnfreezing(["encoder_top", "encoder_bottom"], start_epoch=2, frequency=1)
    restored_trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=2,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={})
    restored.load_state_dict(callback.state_dict())

    assert restored.unfrozen_group_count == 1
    assert restored_trainer.model.encoder_top.weight.requires_grad is True
    assert restored_trainer.model.encoder_bottom.weight.requires_grad is False


def test_lookahead_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="sync_period"):
        Lookahead(sync_period=0)

    with pytest.raises(ValueError, match="slow_step_size"):
        Lookahead(slow_step_size=0.0)

    with pytest.raises(ValueError, match="slow_step_size"):
        Lookahead(slow_step_size=1.5)


def test_lookahead_periodically_syncs_slow_weights_back_into_model():
    callback = Lookahead(sync_period=2, slow_step_size=0.5)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=4,
        callbacks=[callback],
    )

    history = trainer.fit([ToyBatch(2.0)])

    assert history["completed_epochs"] == 4
    assert callback.step_count == 4
    assert torch.allclose(callback.slow_state["weight"], torch.tensor([1.21875]))
    assert torch.allclose(trainer.model.weight.detach(), torch.tensor([1.21875]))


def test_lookahead_state_dict_round_trip_preserves_slow_state():
    callback = Lookahead(sync_period=2, slow_step_size=0.5)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.25,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([ToyBatch(2.0)])
    restored = Lookahead(sync_period=2, slow_step_size=0.5)
    restored.load_state_dict(callback.state_dict())

    assert restored.step_count == 3
    assert torch.allclose(restored.slow_state["weight"], callback.slow_state["weight"])


def test_stochastic_weight_averaging_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_epoch"):
        StochasticWeightAveraging(start_epoch=0)

    with pytest.raises(ValueError, match="frequency"):
        StochasticWeightAveraging(frequency=0)


def test_stochastic_weight_averaging_can_apply_averaged_weights_at_fit_end():
    callback = StochasticWeightAveraging(start_epoch=2, frequency=1, apply_on_fit_end=True)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=3,
        callbacks=[callback],
    )

    history = trainer.fit([ToyBatch(2.0)])

    assert history["completed_epochs"] == 3
    assert callback.num_averaged == 2
    assert torch.allclose(callback.avg_state["weight"], torch.tensor([2.0]))
    assert torch.allclose(trainer.model.weight.detach(), torch.tensor([2.0]))
