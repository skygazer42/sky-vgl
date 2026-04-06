import pytest
import torch
from torch import nn

from vgl.engine import (
    AdaptiveGradientClipping,
    BootstrapBetaScheduler,
    Callback,
    ConfidencePenaltyScheduler,
    EarlyStopping,
    ExponentialMovingAverage,
    FocalGammaScheduler,
    FloodingLevelScheduler,
    GeneralizedCrossEntropyScheduler,
    GradientAccumulationScheduler,
    GradientNoiseInjection,
    GradientValueClipping,
    GradientCentralization,
    GradualUnfreezing,
    HistoryLogger,
    LabelSmoothingScheduler,
    LdamMarginScheduler,
    LogitAdjustTauScheduler,
    Lookahead,
    ModelCheckpoint,
    Poly1EpsilonScheduler,
    PosWeightScheduler,
    StopTraining,
    StochasticWeightAveraging,
    SymmetricCrossEntropyBetaScheduler,
    Trainer,
    WeightDecayScheduler,
)
from vgl.train.task import Task
from vgl.train.tasks import BootstrapTask
from vgl.train.tasks import ConfidencePenaltyTask
from vgl.train.tasks import FloodingTask
from vgl.train.tasks import GeneralizedCrossEntropyTask
from vgl.train.tasks import GraphClassificationTask
from vgl.train.tasks import LinkPredictionTask
from vgl.train.tasks import Poly1CrossEntropyTask
from vgl.train.tasks import SymmetricCrossEntropyTask


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


class RaiseOnFirstEpoch(Callback):
    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del trainer, train_summary, val_summary, history
        if epoch == 1:
            raise RuntimeError("boom")


class StepRecordingCallback(Callback):
    def __init__(self):
        self.steps = []

    def on_before_optimizer_step(self, trainer, step):
        del trainer
        self.steps.append(("before", step))

    def on_after_optimizer_step(self, trainer, step):
        del trainer
        self.steps.append(("after", step))


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class GradientHolderModel(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.weight = nn.Parameter(parameter)


class DummyTrainer:
    def __init__(self, model):
        self.model = model


class MulticlassToyBatch:
    def __init__(self, label):
        self.labels = torch.tensor([label])
        self.metadata = [{"label": label}]


class MulticlassToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[0.5, -0.5]], dtype=torch.float32))

    def forward(self, batch):
        return self.logits.repeat(batch.labels.size(0), 1)


class BinaryToyBatch:
    def __init__(self, labels):
        self.labels = torch.tensor(labels, dtype=torch.float32)


class BinaryToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))

    def forward(self, batch):
        return self.logit.repeat(batch.labels.size(0))


class LabelSmoothingRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while hasattr(task, "base_task"):
            task = task.base_task
        self.values.append(float(task.label_smoothing))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class FocalGammaRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while hasattr(task, "base_task"):
            task = task.base_task
        self.values.append(float(task.focal_gamma))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class LogitAdjustTauRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while hasattr(task, "base_task"):
            task = task.base_task
        self.values.append(float(task.logit_adjust_tau))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class LdamMarginRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while hasattr(task, "base_task"):
            task = task.base_task
        self.values.append(float(task.ldam_max_margin))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class PosWeightRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while hasattr(task, "base_task"):
            task = task.base_task
        self.values.append(float(task.pos_weight.reshape(-1)[0]))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class BootstrapBetaRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while not hasattr(task, "beta"):
            if not hasattr(task, "base_task"):
                raise AssertionError("Bootstrap beta task not found")
            task = task.base_task
        self.values.append(float(task.beta))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class ConfidencePenaltyRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while not hasattr(task, "coefficient"):
            if not hasattr(task, "base_task"):
                raise AssertionError("Confidence penalty task not found")
            task = task.base_task
        self.values.append(float(task.coefficient))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class FloodingLevelRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while not hasattr(task, "level"):
            if not hasattr(task, "base_task"):
                raise AssertionError("Flooding task not found")
            task = task.base_task
        self.values.append(float(task.level))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class GeneralizedCrossEntropyRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while not hasattr(task, "q"):
            if not hasattr(task, "base_task"):
                raise AssertionError("Generalized cross entropy task not found")
            task = task.base_task
        self.values.append(float(task.q))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class Poly1EpsilonRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while not hasattr(task, "epsilon"):
            if not hasattr(task, "base_task"):
                raise AssertionError("Poly1 cross entropy task not found")
            task = task.base_task
        self.values.append(float(task.epsilon))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class SymmetricCrossEntropyBetaRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        task = trainer.task
        while not hasattr(task, "beta"):
            if not hasattr(task, "base_task"):
                raise AssertionError("Symmetric cross entropy task not found")
            task = task.base_task
        self.values.append(float(task.beta))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


class WeightDecayRecorder(Callback):
    def __init__(self):
        self.values = []

    def _record(self, trainer):
        self.values.append(float(trainer.optimizer.param_groups[0]["weight_decay"]))

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


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
            trainer.global_step += 1
            return {"loss": next(train_losses)}
        return {"loss": next(val_losses)}

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)

    history = trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    assert history["completed_epochs"] == 2
    assert logger.records == [
        {
            "epoch": 1,
            "train": {"loss": 1.0},
            "val": {"loss": 2.0},
            "best_epoch": 1,
            "best_metric": 2.0,
            "monitor": "val_loss",
            "global_step": 1,
            "elapsed_seconds": pytest.approx(0.0, abs=1.0),
        },
        {
            "epoch": 2,
            "train": {"loss": 0.5},
            "val": {"loss": 1.5},
            "best_epoch": 2,
            "best_metric": 1.5,
            "monitor": "val_loss",
            "global_step": 2,
            "elapsed_seconds": pytest.approx(0.0, abs=1.0),
        },
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


def test_gradient_value_clipping_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="clip_value"):
        GradientValueClipping(clip_value=0.0)


def test_gradient_centralization_rejects_invalid_configuration():
    with pytest.raises(TypeError, match="conv_only"):
        GradientCentralization(conv_only="yes")


def test_gradient_accumulation_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="scheduling must not be empty"):
        GradientAccumulationScheduler({})

    with pytest.raises(TypeError, match="epoch keys"):
        GradientAccumulationScheduler({"1": 2})

    with pytest.raises(ValueError, match="epoch keys must be >= 1"):
        GradientAccumulationScheduler({0: 2})

    with pytest.raises(TypeError, match="accumulation values"):
        GradientAccumulationScheduler({1: 1.5})

    with pytest.raises(ValueError, match="accumulation values must be >= 1"):
        GradientAccumulationScheduler({1: 0})


def test_gradient_accumulation_scheduler_accepts_tensor_schedule_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("GradientAccumulationScheduler scheduling should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GradientAccumulationScheduler(
        [
            (torch.tensor(1), torch.tensor(1)),
            (torch.tensor(2), torch.tensor(2)),
        ]
    )

    assert callback.scheduling == ((1, 1), (2, 2))


def test_gradient_accumulation_scheduler_updates_epoch_accumulation_and_restores_original_value():
    callback = GradientAccumulationScheduler({1: 1, 2: 2})
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=CountingSGD,
        lr=1.0,
        max_epochs=2,
        accumulate_grad_batches=3,
        callbacks=[callback],
    )

    history = trainer.fit([ToyBatch(1.0), ToyBatch(2.0), ToyBatch(3.0), ToyBatch(4.0)])

    assert history["completed_epochs"] == 2
    assert trainer.optimizer.step_calls == 6
    assert trainer.global_step == 6
    assert callback.current_accumulate_grad_batches == 2
    assert trainer.accumulate_grad_batches == 3


def test_gradient_accumulation_scheduler_on_fit_start_accepts_tensor_state_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("GradientAccumulationScheduler fit state should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GradientAccumulationScheduler({1: 1, 2: 2})
    trainer = DummyTrainer(ToyModel())
    trainer.accumulate_grad_batches = torch.tensor(3)

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(1)})

    assert callback.original_accumulate_grad_batches == 3
    assert callback.current_accumulate_grad_batches == 2
    assert trainer.accumulate_grad_batches == 2


def test_gradient_accumulation_scheduler_load_state_dict_accepts_tensor_state_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("GradientAccumulationScheduler state loading should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GradientAccumulationScheduler({1: 1, 2: 2})
    trainer = DummyTrainer(ToyModel())
    trainer.accumulate_grad_batches = 1
    callback._trainer = trainer

    callback.load_state_dict(
        {
            "original_accumulate_grad_batches": torch.tensor(3),
            "current_accumulate_grad_batches": torch.tensor(2),
        }
    )

    assert callback.original_accumulate_grad_batches == 3
    assert callback.current_accumulate_grad_batches == 2
    assert trainer.accumulate_grad_batches == 2


def test_model_checkpoint_rejects_invalid_configuration(tmp_path):
    with pytest.raises(ValueError, match="filename"):
        ModelCheckpoint(tmp_path, filename="")

    with pytest.raises(ValueError, match="mode"):
        ModelCheckpoint(tmp_path, mode="median")

    with pytest.raises(TypeError, match="save_top_k"):
        ModelCheckpoint(tmp_path, save_top_k=1.5)

    with pytest.raises(ValueError, match="save_top_k"):
        ModelCheckpoint(tmp_path, save_top_k=-2)

    with pytest.raises(TypeError, match="save_last"):
        ModelCheckpoint(tmp_path, save_last="yes")

    with pytest.raises(TypeError, match="save_on_exception"):
        ModelCheckpoint(tmp_path, save_on_exception="yes")

    with pytest.raises(TypeError, match="every_n_epochs"):
        ModelCheckpoint(tmp_path, every_n_epochs=1.5)

    with pytest.raises(ValueError, match="every_n_epochs"):
        ModelCheckpoint(tmp_path, every_n_epochs=0)


def test_model_checkpoint_accepts_tensor_scalar_counts_without_tensor_int(tmp_path, monkeypatch):
    def fail_int(self):
        raise AssertionError("ModelCheckpoint count parameters should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = ModelCheckpoint(
        tmp_path,
        save_top_k=torch.tensor(1),
        every_n_epochs=torch.tensor(2),
    )

    assert callback.save_top_k == 1
    assert callback.every_n_epochs == 2


def test_model_checkpoint_saves_top_k_and_last(tmp_path):
    callback = ModelCheckpoint(
        tmp_path / "checkpoints",
        filename="epoch{epoch}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=3,
        monitor="val_loss",
        callbacks=[callback],
    )

    history = trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])

    assert history["completed_epochs"] == 3
    assert callback.best_model_path is not None
    assert callback.best_model_path.endswith("epoch2.ckpt")
    assert callback.best_model_score == pytest.approx(1.0)
    assert callback.kth_best_model_score == pytest.approx(1.0)
    assert callback.last_model_path is not None
    assert callback.last_model_path.endswith("last.ckpt")
    assert (tmp_path / "checkpoints" / "epoch2.ckpt").exists()
    assert not (tmp_path / "checkpoints" / "epoch1.ckpt").exists()
    assert (tmp_path / "checkpoints" / "last.ckpt").exists()
    payload = Trainer.load_checkpoint(tmp_path / "checkpoints" / "last.ckpt")
    assert payload["metadata"]["epoch"] == 3
    assert payload["metadata"]["global_step"] == 3
    assert payload["metadata"]["tag"] == "last"


def test_model_checkpoint_can_save_checkpoint_on_exception(tmp_path):
    callback = ModelCheckpoint(
        tmp_path / "checkpoints",
        save_top_k=0,
        save_last=False,
        save_on_exception=True,
    )
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=2,
        callbacks=[callback, RaiseOnFirstEpoch()],
    )

    with pytest.raises(RuntimeError, match="boom"):
        trainer.fit([ToyBatch(1.0)])

    exception_path = tmp_path / "checkpoints" / "exception.ckpt"
    assert exception_path.exists()
    assert callback.exception_model_path == str(exception_path)
    payload = Trainer.load_checkpoint(exception_path)
    assert payload["metadata"]["tag"] == "exception"
    assert payload["metadata"]["exception_type"] == "RuntimeError"
    assert payload["metadata"]["exception_message"] == "boom"


def test_gradient_value_clipping_clips_dense_gradients():
    callback = GradientValueClipping(clip_value=0.5)
    model = GradientHolderModel(torch.zeros(2, 2))
    model.weight.grad = torch.tensor([[1.5, -1.2], [0.25, -0.4]])
    trainer = DummyTrainer(model)

    callback.on_before_optimizer_step(trainer, step=1)

    assert torch.allclose(
        model.weight.grad,
        torch.tensor([[0.5, -0.5], [0.25, -0.4]]),
    )


def test_gradient_value_clipping_skips_sparse_gradients():
    callback = GradientValueClipping(clip_value=0.5)
    model = GradientHolderModel(torch.zeros(2, 2))
    sparse_grad = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.tensor([1.5, -1.2]),
        size=(2, 2),
    )
    model.weight.grad = sparse_grad
    trainer = DummyTrainer(model)

    callback.on_before_optimizer_step(trainer, step=1)

    assert model.weight.grad is sparse_grad


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


def test_gradual_unfreezing_accepts_tensor_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("GradualUnfreezing config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GradualUnfreezing(
        ["encoder"],
        start_epoch=torch.tensor(2),
        frequency=torch.tensor(3),
    )

    assert callback.start_epoch == 2
    assert callback.frequency == 3


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


def test_gradual_unfreezing_load_state_dict_accepts_tensor_state_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("GradualUnfreezing state should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GradualUnfreezing(["encoder"])
    callback.group_param_names = (("encoder.weight",), ("head.weight",))

    callback.load_state_dict(
        {
            "original_requires_grad": {"encoder.weight": True},
            "unfrozen_group_count": torch.tensor(1),
        }
    )

    assert callback.unfrozen_group_count == 1
    assert callback.original_requires_grad == {"encoder.weight": True}


def test_lookahead_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="sync_period"):
        Lookahead(sync_period=0)

    with pytest.raises(ValueError, match="slow_step_size"):
        Lookahead(slow_step_size=0.0)

    with pytest.raises(ValueError, match="slow_step_size"):
        Lookahead(slow_step_size=1.5)


def test_lookahead_accepts_tensor_sync_period_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("Lookahead sync_period should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = Lookahead(sync_period=torch.tensor(2), slow_step_size=0.5)

    assert callback.sync_period == 2
    assert callback.slow_step_size == pytest.approx(0.5)


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


def test_lookahead_load_state_dict_accepts_tensor_step_count_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("Lookahead state loading should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = Lookahead(sync_period=2, slow_step_size=0.5)
    callback.load_state_dict(
        {
            "slow_state": {"weight": torch.tensor([1.0])},
            "step_count": torch.tensor(3),
        }
    )

    assert callback.step_count == 3
    assert torch.allclose(callback.slow_state["weight"], torch.tensor([1.0]))


def test_lookahead_on_after_optimizer_step_accepts_tensor_step_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("Lookahead step handling should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = Lookahead(sync_period=2, slow_step_size=0.5)
    callback.slow_state = {"weight": torch.tensor([0.0])}
    trainer = DummyTrainer(ToyModel())

    callback.on_after_optimizer_step(trainer, step=torch.tensor(1))

    assert callback.step_count == 1
    assert torch.allclose(callback.slow_state["weight"], torch.tensor([0.0]))


def test_stochastic_weight_averaging_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_epoch"):
        StochasticWeightAveraging(start_epoch=0)

    with pytest.raises(ValueError, match="frequency"):
        StochasticWeightAveraging(frequency=0)


def test_stochastic_weight_averaging_accepts_tensor_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("StochasticWeightAveraging config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = StochasticWeightAveraging(
        start_epoch=torch.tensor(2),
        frequency=torch.tensor(3),
    )

    assert callback.start_epoch == 2
    assert callback.frequency == 3


def test_stochastic_weight_averaging_load_state_dict_accepts_tensor_state_without_tensor_int(
    monkeypatch,
):
    def fail_int(self):
        raise AssertionError("StochasticWeightAveraging state should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = StochasticWeightAveraging()

    callback.load_state_dict(
        {
            "avg_state": {"weight": torch.tensor([2.0])},
            "num_averaged": torch.tensor(3),
        }
    )

    assert callback.num_averaged == 3
    assert torch.allclose(callback.avg_state["weight"], torch.tensor([2.0]))


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


def test_label_smoothing_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        LabelSmoothingScheduler(start_value=-0.1, end_value=0.2, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        LabelSmoothingScheduler(start_value=0.0, end_value=1.0, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=3, end_epoch=2)


def test_label_smoothing_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("LabelSmoothingScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = LabelSmoothingScheduler(
        start_value=0.0,
        end_value=0.2,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_label_smoothing_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("LabelSmoothingScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(0.1)


def test_label_smoothing_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("LabelSmoothingScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=2, end_epoch=4)
    task = GraphClassificationTask(target="label", label_source="graph", label_smoothing=0.05)
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(0.1)
    assert task.label_smoothing == pytest.approx(0.1)


def test_label_smoothing_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=2, end_epoch=4)
    recorder = LabelSmoothingRecorder()
    task = GraphClassificationTask(target="label", label_source="graph", label_smoothing=0.05)
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.0, 0.0, 0.1, 0.2, 0.2])
    assert callback.current_value == pytest.approx(0.2)
    assert task.label_smoothing == pytest.approx(0.05)


def test_label_smoothing_scheduler_state_dict_round_trip_preserves_current_value():
    callback = LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=2, end_epoch=4)
    task = GraphClassificationTask(target="label", label_source="graph", label_smoothing=0.05)
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = LabelSmoothingScheduler(start_value=0.0, end_value=0.2, start_epoch=2, end_epoch=4)
    restored_task = GraphClassificationTask(target="label", label_source="graph", label_smoothing=0.05)
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(0.1)
    assert restored_task.label_smoothing == pytest.approx(0.1)


def test_label_smoothing_scheduler_requires_task_with_label_smoothing():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[LabelSmoothingScheduler(start_value=0.0, end_value=0.1, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="label_smoothing"):
        trainer.fit([ToyBatch(1.0)])


def test_focal_gamma_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        FocalGammaScheduler(start_value=-0.1, end_value=3.0, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        FocalGammaScheduler(start_value=0.0, end_value=-0.1, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        FocalGammaScheduler(start_value=0.0, end_value=3.0, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        FocalGammaScheduler(start_value=0.0, end_value=3.0, start_epoch=3, end_epoch=2)


def test_focal_gamma_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("FocalGammaScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = FocalGammaScheduler(
        start_value=0.5,
        end_value=3.0,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_focal_gamma_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("FocalGammaScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = FocalGammaScheduler(start_value=0.5, end_value=3.0, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(1.75)


def test_focal_gamma_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("FocalGammaScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = FocalGammaScheduler(start_value=0.5, end_value=3.0, start_epoch=2, end_epoch=4)
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="focal",
        focal_gamma=1.5,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(1.75)
    assert task.focal_gamma == pytest.approx(1.75)


def test_focal_gamma_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = FocalGammaScheduler(start_value=0.5, end_value=3.0, start_epoch=2, end_epoch=4)
    recorder = FocalGammaRecorder()
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="focal",
        focal_gamma=1.5,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.5, 0.5, 1.75, 3.0, 3.0])
    assert callback.current_value == pytest.approx(3.0)
    assert task.focal_gamma == pytest.approx(1.5)


def test_focal_gamma_scheduler_state_dict_round_trip_preserves_current_value():
    callback = FocalGammaScheduler(start_value=0.5, end_value=3.0, start_epoch=2, end_epoch=4)
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="focal",
        focal_gamma=1.5,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = FocalGammaScheduler(start_value=0.5, end_value=3.0, start_epoch=2, end_epoch=4)
    restored_task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="focal",
        focal_gamma=1.5,
    )
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(1.75)
    assert restored_task.focal_gamma == pytest.approx(1.75)


def test_focal_gamma_scheduler_requires_task_with_focal_gamma():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[FocalGammaScheduler(start_value=0.0, end_value=1.0, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="focal_gamma"):
        trainer.fit([ToyBatch(1.0)])


def test_logit_adjust_tau_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        LogitAdjustTauScheduler(start_value=-0.1, end_value=1.5, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        LogitAdjustTauScheduler(start_value=0.0, end_value=-0.1, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=3, end_epoch=2)


def test_logit_adjust_tau_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("LogitAdjustTauScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = LogitAdjustTauScheduler(
        start_value=0.0,
        end_value=1.5,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_logit_adjust_tau_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("LogitAdjustTauScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(0.75)


def test_logit_adjust_tau_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("LogitAdjustTauScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=2, end_epoch=4)
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="logit_adjustment",
        class_count=[3.0, 1.0],
        logit_adjust_tau=0.2,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(0.75)
    assert task.logit_adjust_tau == pytest.approx(0.75)


def test_logit_adjust_tau_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=2, end_epoch=4)
    recorder = LogitAdjustTauRecorder()
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="logit_adjustment",
        class_count=[3.0, 1.0],
        logit_adjust_tau=0.2,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.0, 0.0, 0.75, 1.5, 1.5])
    assert callback.current_value == pytest.approx(1.5)
    assert task.logit_adjust_tau == pytest.approx(0.2)


def test_logit_adjust_tau_scheduler_state_dict_round_trip_preserves_current_value():
    callback = LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=2, end_epoch=4)
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="logit_adjustment",
        class_count=[3.0, 1.0],
        logit_adjust_tau=0.2,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = LogitAdjustTauScheduler(start_value=0.0, end_value=1.5, start_epoch=2, end_epoch=4)
    restored_task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="logit_adjustment",
        class_count=[3.0, 1.0],
        logit_adjust_tau=0.2,
    )
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(0.75)
    assert restored_task.logit_adjust_tau == pytest.approx(0.75)


def test_logit_adjust_tau_scheduler_requires_task_with_logit_adjust_tau():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[LogitAdjustTauScheduler(start_value=0.0, end_value=1.0, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="logit_adjust_tau"):
        trainer.fit([ToyBatch(1.0)])


def test_ldam_margin_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        LdamMarginScheduler(start_value=0.0, end_value=0.4, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        LdamMarginScheduler(start_value=0.2, end_value=0.0, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        LdamMarginScheduler(start_value=0.2, end_value=0.4, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        LdamMarginScheduler(start_value=0.2, end_value=0.4, start_epoch=3, end_epoch=2)


def test_ldam_margin_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("LdamMarginScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = LdamMarginScheduler(
        start_value=0.2,
        end_value=0.5,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_ldam_margin_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("LdamMarginScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = LdamMarginScheduler(start_value=0.2, end_value=0.5, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(0.35)


def test_ldam_margin_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("LdamMarginScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = LdamMarginScheduler(start_value=0.2, end_value=0.5, start_epoch=2, end_epoch=4)
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="ldam",
        class_count=[3.0, 1.0],
        ldam_max_margin=0.35,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(0.35)
    assert task.ldam_max_margin == pytest.approx(0.35)


def test_ldam_margin_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = LdamMarginScheduler(start_value=0.2, end_value=0.5, start_epoch=2, end_epoch=4)
    recorder = LdamMarginRecorder()
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="ldam",
        class_count=[3.0, 1.0],
        ldam_max_margin=0.35,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.2, 0.2, 0.35, 0.5, 0.5])
    assert callback.current_value == pytest.approx(0.5)
    assert task.ldam_max_margin == pytest.approx(0.35)


def test_ldam_margin_scheduler_state_dict_round_trip_preserves_current_value():
    callback = LdamMarginScheduler(start_value=0.2, end_value=0.5, start_epoch=2, end_epoch=4)
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="ldam",
        class_count=[3.0, 1.0],
        ldam_max_margin=0.35,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = LdamMarginScheduler(start_value=0.2, end_value=0.5, start_epoch=2, end_epoch=4)
    restored_task = GraphClassificationTask(
        target="label",
        label_source="graph",
        loss="ldam",
        class_count=[3.0, 1.0],
        ldam_max_margin=0.35,
    )
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(0.35)
    assert restored_task.ldam_max_margin == pytest.approx(0.35)


def test_ldam_margin_scheduler_requires_task_with_ldam_max_margin():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[LdamMarginScheduler(start_value=0.2, end_value=0.5, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="ldam_max_margin"):
        trainer.fit([ToyBatch(1.0)])


def test_pos_weight_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        PosWeightScheduler(start_value=0.0, end_value=4.0, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        PosWeightScheduler(start_value=1.0, end_value=0.0, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=3, end_epoch=2)


def test_pos_weight_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("PosWeightScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = PosWeightScheduler(
        start_value=1.0,
        end_value=4.0,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_pos_weight_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("PosWeightScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(2.5)


def test_pos_weight_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("PosWeightScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=2, end_epoch=4)
    task = LinkPredictionTask(target="label", pos_weight=2.0)
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(2.5)
    assert torch.allclose(task.pos_weight, torch.tensor([2.5]))


def test_pos_weight_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=2, end_epoch=4)
    recorder = PosWeightRecorder()
    task = LinkPredictionTask(target="label", pos_weight=2.0)
    trainer = Trainer(
        model=BinaryToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([BinaryToyBatch([1.0, 0.0, 1.0])])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([1.0, 1.0, 2.5, 4.0, 4.0])
    assert callback.current_value == pytest.approx(4.0)
    assert torch.allclose(task.pos_weight, torch.tensor([2.0]))


def test_pos_weight_scheduler_state_dict_round_trip_preserves_current_value():
    callback = PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=2, end_epoch=4)
    task = LinkPredictionTask(target="label", pos_weight=2.0)
    trainer = Trainer(
        model=BinaryToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([BinaryToyBatch([1.0, 0.0, 1.0])])

    restored = PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=2, end_epoch=4)
    restored_task = LinkPredictionTask(target="label", pos_weight=2.0)
    restored_trainer = Trainer(
        model=BinaryToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(2.5)
    assert torch.allclose(restored_task.pos_weight, torch.tensor([2.5]))


def test_pos_weight_scheduler_requires_task_with_pos_weight():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[PosWeightScheduler(start_value=1.0, end_value=4.0, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="pos_weight"):
        trainer.fit([ToyBatch(1.0)])


def test_bootstrap_beta_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        BootstrapBetaScheduler(start_value=-0.1, end_value=0.9, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        BootstrapBetaScheduler(start_value=0.2, end_value=1.1, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        BootstrapBetaScheduler(start_value=0.2, end_value=0.9, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        BootstrapBetaScheduler(start_value=0.2, end_value=0.9, start_epoch=3, end_epoch=2)


def test_bootstrap_beta_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("BootstrapBetaScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = BootstrapBetaScheduler(
        start_value=0.2,
        end_value=0.8,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_bootstrap_beta_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("BootstrapBetaScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = BootstrapBetaScheduler(start_value=0.2, end_value=0.8, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(0.5)


def test_bootstrap_beta_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("BootstrapBetaScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = BootstrapBetaScheduler(start_value=0.2, end_value=0.8, start_epoch=2, end_epoch=4)
    task = BootstrapTask(
        GraphClassificationTask(target="label", label_source="graph"),
        beta=0.95,
        mode="soft",
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(0.5)
    assert task.beta == pytest.approx(0.5)


def test_bootstrap_beta_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = BootstrapBetaScheduler(start_value=0.2, end_value=0.8, start_epoch=2, end_epoch=4)
    recorder = BootstrapBetaRecorder()
    task = BootstrapTask(
        GraphClassificationTask(target="label", label_source="graph"),
        beta=0.95,
        mode="soft",
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.2, 0.2, 0.5, 0.8, 0.8])
    assert callback.current_value == pytest.approx(0.8)
    assert task.beta == pytest.approx(0.95)


def test_bootstrap_beta_scheduler_state_dict_round_trip_preserves_current_value():
    callback = BootstrapBetaScheduler(start_value=0.2, end_value=0.8, start_epoch=2, end_epoch=4)
    task = BootstrapTask(
        GraphClassificationTask(target="label", label_source="graph"),
        beta=0.95,
        mode="soft",
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = BootstrapBetaScheduler(start_value=0.2, end_value=0.8, start_epoch=2, end_epoch=4)
    restored_task = BootstrapTask(
        GraphClassificationTask(target="label", label_source="graph"),
        beta=0.95,
        mode="soft",
    )
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(0.5)
    assert restored_task.beta == pytest.approx(0.5)


def test_bootstrap_beta_scheduler_requires_task_with_beta():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[BootstrapBetaScheduler(start_value=0.2, end_value=0.8, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="beta"):
        trainer.fit([ToyBatch(1.0)])


def test_confidence_penalty_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        ConfidencePenaltyScheduler(start_value=-0.1, end_value=0.3, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        ConfidencePenaltyScheduler(start_value=0.0, end_value=-0.1, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=3, end_epoch=2)


def test_confidence_penalty_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("ConfidencePenaltyScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = ConfidencePenaltyScheduler(
        start_value=0.0,
        end_value=0.3,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_confidence_penalty_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("ConfidencePenaltyScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(0.15)


def test_confidence_penalty_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("ConfidencePenaltyScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    task = ConfidencePenaltyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        coefficient=0.1,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(0.15)
    assert task.coefficient == pytest.approx(0.15)


def test_confidence_penalty_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    recorder = ConfidencePenaltyRecorder()
    task = ConfidencePenaltyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        coefficient=0.1,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.0, 0.0, 0.15, 0.3, 0.3])
    assert callback.current_value == pytest.approx(0.3)
    assert task.coefficient == pytest.approx(0.1)


def test_confidence_penalty_scheduler_state_dict_round_trip_preserves_current_value():
    callback = ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    task = ConfidencePenaltyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        coefficient=0.1,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    restored_task = ConfidencePenaltyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        coefficient=0.1,
    )
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(0.15)
    assert restored_task.coefficient == pytest.approx(0.15)


def test_confidence_penalty_scheduler_requires_task_with_coefficient():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[ConfidencePenaltyScheduler(start_value=0.0, end_value=0.3, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="coefficient"):
        trainer.fit([ToyBatch(1.0)])


def test_flooding_level_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        FloodingLevelScheduler(start_value=-0.1, end_value=0.3, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        FloodingLevelScheduler(start_value=0.0, end_value=-0.1, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=3, end_epoch=2)


def test_flooding_level_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("FloodingLevelScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = FloodingLevelScheduler(
        start_value=0.0,
        end_value=0.3,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_flooding_level_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("FloodingLevelScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(0.15)


def test_flooding_level_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("FloodingLevelScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    task = FloodingTask(
        GraphClassificationTask(target="label", label_source="graph"),
        level=0.1,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(0.15)
    assert task.level == pytest.approx(0.15)


def test_flooding_level_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    recorder = FloodingLevelRecorder()
    task = FloodingTask(
        GraphClassificationTask(target="label", label_source="graph"),
        level=0.1,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.0, 0.0, 0.15, 0.3, 0.3])
    assert callback.current_value == pytest.approx(0.3)
    assert task.level == pytest.approx(0.1)


def test_flooding_level_scheduler_state_dict_round_trip_preserves_current_value():
    callback = FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    task = FloodingTask(
        GraphClassificationTask(target="label", label_source="graph"),
        level=0.1,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=2, end_epoch=4)
    restored_task = FloodingTask(
        GraphClassificationTask(target="label", label_source="graph"),
        level=0.1,
    )
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(0.15)
    assert restored_task.level == pytest.approx(0.15)


def test_flooding_level_scheduler_requires_task_with_level():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[FloodingLevelScheduler(start_value=0.0, end_value=0.3, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="level"):
        trainer.fit([ToyBatch(1.0)])


def test_generalized_cross_entropy_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        GeneralizedCrossEntropyScheduler(start_value=0.0, end_value=0.9, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=1.1, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=3, end_epoch=2)


def test_generalized_cross_entropy_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(
    monkeypatch,
):
    def fail_int(self):
        raise AssertionError("GeneralizedCrossEntropyScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GeneralizedCrossEntropyScheduler(
        start_value=0.3,
        end_value=0.9,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_generalized_cross_entropy_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(
    monkeypatch,
):
    def fail_int(self):
        raise AssertionError("GeneralizedCrossEntropyScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(0.6)


def test_generalized_cross_entropy_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(
    monkeypatch,
):
    def fail_int(self):
        raise AssertionError("GeneralizedCrossEntropyScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    task = GeneralizedCrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        q=0.7,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(0.6)
    assert task.q == pytest.approx(0.6)


def test_generalized_cross_entropy_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    recorder = GeneralizedCrossEntropyRecorder()
    task = GeneralizedCrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        q=0.7,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.3, 0.3, 0.6, 0.9, 0.9])
    assert callback.current_value == pytest.approx(0.9)
    assert task.q == pytest.approx(0.7)


def test_generalized_cross_entropy_scheduler_state_dict_round_trip_preserves_current_value():
    callback = GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    task = GeneralizedCrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        q=0.7,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    restored_task = GeneralizedCrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        q=0.7,
    )
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(0.6)
    assert restored_task.q == pytest.approx(0.6)


def test_generalized_cross_entropy_scheduler_requires_task_with_q():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[GeneralizedCrossEntropyScheduler(start_value=0.3, end_value=0.9, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="q"):
        trainer.fit([ToyBatch(1.0)])


def test_poly1_epsilon_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        Poly1EpsilonScheduler(start_value=-0.1, end_value=0.9, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        Poly1EpsilonScheduler(start_value=0.3, end_value=-0.1, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=3, end_epoch=2)


def test_poly1_epsilon_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("Poly1EpsilonScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = Poly1EpsilonScheduler(
        start_value=0.3,
        end_value=0.9,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_poly1_epsilon_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("Poly1EpsilonScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(0.6)


def test_poly1_epsilon_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("Poly1EpsilonScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    task = Poly1CrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        epsilon=0.7,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_value == pytest.approx(0.6)
    assert task.epsilon == pytest.approx(0.6)


def test_poly1_epsilon_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    recorder = Poly1EpsilonRecorder()
    task = Poly1CrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        epsilon=0.7,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.3, 0.3, 0.6, 0.9, 0.9])
    assert callback.current_value == pytest.approx(0.9)
    assert task.epsilon == pytest.approx(0.7)


def test_poly1_epsilon_scheduler_state_dict_round_trip_preserves_current_value():
    callback = Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    task = Poly1CrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        epsilon=0.7,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    restored_task = Poly1CrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        epsilon=0.7,
    )
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(0.6)
    assert restored_task.epsilon == pytest.approx(0.6)


def test_poly1_epsilon_scheduler_requires_task_with_epsilon():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[Poly1EpsilonScheduler(start_value=0.3, end_value=0.9, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="epsilon"):
        trainer.fit([ToyBatch(1.0)])


def test_symmetric_cross_entropy_beta_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_value"):
        SymmetricCrossEntropyBetaScheduler(start_value=-0.1, end_value=0.9, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_value"):
        SymmetricCrossEntropyBetaScheduler(start_value=0.3, end_value=-0.1, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        SymmetricCrossEntropyBetaScheduler(start_value=0.3, end_value=0.9, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        SymmetricCrossEntropyBetaScheduler(start_value=0.3, end_value=0.9, start_epoch=3, end_epoch=2)


def test_symmetric_cross_entropy_beta_scheduler_updates_task_on_linear_schedule_and_restores_original_value():
    callback = SymmetricCrossEntropyBetaScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    recorder = SymmetricCrossEntropyBetaRecorder()
    task = SymmetricCrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        alpha=1.0,
        beta=0.7,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([MulticlassToyBatch(0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.3, 0.3, 0.6, 0.9, 0.9])
    assert callback.current_value == pytest.approx(0.9)
    assert task.beta == pytest.approx(0.7)


def test_symmetric_cross_entropy_beta_scheduler_state_dict_round_trip_preserves_current_value():
    callback = SymmetricCrossEntropyBetaScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    task = SymmetricCrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        alpha=1.0,
        beta=0.7,
    )
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([MulticlassToyBatch(0)])

    restored = SymmetricCrossEntropyBetaScheduler(start_value=0.3, end_value=0.9, start_epoch=2, end_epoch=4)
    restored_task = SymmetricCrossEntropyTask(
        GraphClassificationTask(target="label", label_source="graph"),
        alpha=1.0,
        beta=0.7,
    )
    restored_trainer = Trainer(
        model=MulticlassToyModel(),
        task=restored_task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_value == pytest.approx(0.6)
    assert restored_task.beta == pytest.approx(0.6)


def test_symmetric_cross_entropy_beta_scheduler_requires_task_with_beta():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        callbacks=[SymmetricCrossEntropyBetaScheduler(start_value=0.3, end_value=0.9, start_epoch=1, end_epoch=2)],
    )

    with pytest.raises(ValueError, match="beta"):
        trainer.fit([ToyBatch(1.0)])


def test_weight_decay_scheduler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_factor"):
        WeightDecayScheduler(start_factor=-0.1, end_factor=1.0, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="end_factor"):
        WeightDecayScheduler(start_factor=0.0, end_factor=-0.1, start_epoch=1, end_epoch=3)

    with pytest.raises(ValueError, match="start_epoch"):
        WeightDecayScheduler(start_factor=0.0, end_factor=1.0, start_epoch=0, end_epoch=3)

    with pytest.raises(ValueError, match="end_epoch"):
        WeightDecayScheduler(start_factor=0.0, end_factor=1.0, start_epoch=3, end_epoch=2)


def test_weight_decay_scheduler_accepts_tensor_epoch_configuration_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("WeightDecayScheduler config should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = WeightDecayScheduler(
        start_factor=0.0,
        end_factor=1.5,
        start_epoch=torch.tensor(2),
        end_epoch=torch.tensor(4),
    )

    assert callback.start_epoch == 2
    assert callback.end_epoch == 4


def test_weight_decay_scheduler_value_for_epoch_accepts_tensor_epoch_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("WeightDecayScheduler epoch math should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = WeightDecayScheduler(start_factor=0.0, end_factor=1.5, start_epoch=2, end_epoch=4)

    assert callback._value_for_epoch(torch.tensor(3)) == pytest.approx(0.75)


def test_weight_decay_scheduler_on_fit_start_accepts_tensor_history_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("WeightDecayScheduler history should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = WeightDecayScheduler(start_factor=0.0, end_factor=1.5, start_epoch=2, end_epoch=4)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=lambda params, lr: torch.optim.SGD(params, lr=lr, weight_decay=0.2),
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    callback.on_fit_start(trainer, history={"completed_epochs": torch.tensor(2)})

    assert callback.current_factor == pytest.approx(0.75)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.15)


def test_weight_decay_scheduler_updates_optimizer_on_linear_schedule_and_restores_original_values():
    callback = WeightDecayScheduler(start_factor=0.0, end_factor=1.5, start_epoch=2, end_epoch=4)
    recorder = WeightDecayRecorder()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=lambda params, lr: torch.optim.SGD(params, lr=lr, weight_decay=0.2),
        lr=0.0,
        max_epochs=4,
        callbacks=[callback, recorder],
    )

    history = trainer.fit([ToyBatch(1.0)])

    assert history["completed_epochs"] == 4
    assert recorder.values == pytest.approx([0.0, 0.0, 0.15, 0.3, 0.3])
    assert callback.current_factor == pytest.approx(1.5)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)


def test_weight_decay_scheduler_state_dict_round_trip_preserves_current_factor():
    callback = WeightDecayScheduler(start_factor=0.0, end_factor=1.5, start_epoch=2, end_epoch=4)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=lambda params, lr: torch.optim.SGD(params, lr=lr, weight_decay=0.2),
        lr=0.0,
        max_epochs=3,
        callbacks=[callback],
    )

    trainer.fit([ToyBatch(1.0)])

    restored = WeightDecayScheduler(start_factor=0.0, end_factor=1.5, start_epoch=2, end_epoch=4)
    restored_trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=lambda params, lr: torch.optim.SGD(params, lr=lr, weight_decay=0.2),
        lr=0.0,
        max_epochs=3,
        callbacks=[restored],
    )
    restored.on_fit_start(restored_trainer, history={"completed_epochs": 0})
    restored.load_state_dict(callback.state_dict())

    assert restored.current_factor == pytest.approx(0.75)
    assert restored_trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.15)


def test_gradient_noise_injection_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="std"):
        GradientNoiseInjection(std=0.0)

    with pytest.raises(ValueError, match="decay_exponent"):
        GradientNoiseInjection(std=0.1, decay_exponent=-0.1)


def test_gradient_noise_injection_accepts_tensor_seed_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("GradientNoiseInjection seed should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GradientNoiseInjection(std=0.1, decay_exponent=0.0, seed=torch.tensor(7))

    assert callback.seed == 7


def test_gradient_noise_injection_adds_deterministic_noise_to_dense_gradients():
    parameter = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)
    model = GradientHolderModel(parameter)
    model.weight.grad = torch.ones_like(model.weight)
    trainer = DummyTrainer(model)
    callback = GradientNoiseInjection(std=0.2, decay_exponent=0.5, seed=13)

    callback.on_fit_start(trainer, history={})
    callback.on_before_optimizer_step(trainer, step=4)

    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(13)
    expected_std = 0.2 / (4 ** 0.5)
    expected_noise = torch.randn(model.weight.grad.shape, generator=expected_generator) * expected_std
    expected_grad = torch.ones_like(model.weight) + expected_noise

    assert torch.allclose(model.weight.grad, expected_grad)
    assert callback.step_count == 4


def test_gradient_noise_injection_state_dict_round_trip_preserves_generator_progress():
    parameter = torch.tensor([1.0, -1.0], dtype=torch.float32)
    trainer = DummyTrainer(GradientHolderModel(parameter))
    trainer.model.weight.grad = torch.zeros_like(trainer.model.weight)
    callback = GradientNoiseInjection(std=0.1, decay_exponent=0.0, seed=7)

    callback.on_fit_start(trainer, history={})
    callback.on_before_optimizer_step(trainer, step=1)
    callback.on_before_optimizer_step(trainer, step=2)

    restored_trainer = DummyTrainer(GradientHolderModel(parameter))
    restored_trainer.model.weight.grad = torch.zeros_like(restored_trainer.model.weight)
    restored = GradientNoiseInjection(std=0.1, decay_exponent=0.0, seed=7)
    restored.on_fit_start(restored_trainer, history={})
    restored.load_state_dict(callback.state_dict())

    trainer.model.weight.grad = torch.zeros_like(trainer.model.weight)
    restored_trainer.model.weight.grad = torch.zeros_like(restored_trainer.model.weight)
    callback.on_before_optimizer_step(trainer, step=3)
    restored.on_before_optimizer_step(restored_trainer, step=3)

    assert restored.step_count == callback.step_count == 3
    assert torch.allclose(restored_trainer.model.weight.grad, trainer.model.weight.grad)


def test_gradient_noise_injection_load_state_dict_accepts_tensor_step_count_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("GradientNoiseInjection state loading should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    callback = GradientNoiseInjection(std=0.1, decay_exponent=0.0, seed=7)
    callback.load_state_dict({"step_count": torch.tensor(2), "generator_state": None})

    assert callback.step_count == 2


def test_gradient_noise_injection_on_before_optimizer_step_accepts_tensor_step_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("GradientNoiseInjection step handling should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    parameter = torch.tensor([1.0, -1.0], dtype=torch.float32)
    trainer = DummyTrainer(GradientHolderModel(parameter))
    trainer.model.weight.grad = torch.zeros_like(trainer.model.weight)
    callback = GradientNoiseInjection(std=0.1, decay_exponent=0.0, seed=7)
    callback.on_fit_start(trainer, history={})
    callback.on_before_optimizer_step(trainer, step=torch.tensor(1))

    assert callback.step_count == 1
