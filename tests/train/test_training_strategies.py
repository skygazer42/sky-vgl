import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vgl.engine import (
    ASAM,
    Callback,
    DeferredReweighting,
    GSAM,
    LayerwiseLrDecay,
    SAM,
    Trainer,
    WarmupCosineScheduler,
)
from vgl.train.task import Task
from vgl.train.tasks import GraphClassificationTask, RDropTask


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


class VectorToyBatch:
    def __init__(self, target):
        self.target = torch.tensor(target, dtype=torch.float32)


class VectorToyModel(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

    def forward(self, batch):
        del batch
        return self.weight


class WeightedVectorToyTask(Task):
    def __init__(self, curvature):
        self.curvature = torch.tensor(curvature, dtype=torch.float32)

    def loss(self, batch, predictions, stage):
        del stage
        return (((predictions - batch.target) ** 2) * self.curvature).sum()

    def targets(self, batch, stage):
        del stage
        return batch.target


class RDropBatch:
    def __init__(self, logits_a, logits_b, label):
        self.logits_a = torch.tensor([logits_a], dtype=torch.float32)
        self.logits_b = torch.tensor([logits_b], dtype=torch.float32)
        self.labels = torch.tensor([label])
        self.metadata = [{"label": label}]


class AlternatingLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        base = batch.logits_a if self.calls % 2 == 1 else batch.logits_b
        return base + self.bias


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


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class LrRecordingCallback(Callback):
    def __init__(self):
        self.lrs = []

    def on_fit_start(self, trainer, history):
        del history
        self.lrs.append(trainer.optimizer.param_groups[0]["lr"])

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self.lrs.append(trainer.optimizer.param_groups[0]["lr"])


class StepLrRecordingCallback(Callback):
    def __init__(self):
        self.lrs = []

    def on_fit_start(self, trainer, history):
        del history
        self.lrs.append(trainer.optimizer.param_groups[0]["lr"])

    def on_after_optimizer_step(self, trainer, step):
        del step
        self.lrs.append(trainer.optimizer.param_groups[0]["lr"])


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


class ClassWeightRecordingCallback(Callback):
    def __init__(self):
        self.weights = []

    def _record(self, trainer):
        task = trainer.task
        while hasattr(task, "base_task"):
            task = task.base_task
        weight = task.class_weight
        self.weights.append(None if weight is None else weight.detach().clone())

    def on_fit_start(self, trainer, history):
        del history
        self._record(trainer)

    def on_epoch_end(self, trainer, epoch, train_summary, val_summary, history):
        del epoch, train_summary, val_summary, history
        self._record(trainer)


def _expected_drw_weight(class_count, *, beta):
    count = torch.tensor(class_count, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(torch.full_like(count, beta), count)
    weight = (1.0 - beta) / effective_num
    return weight / weight.sum() * weight.numel()


class NoOpGradScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        return None


def test_trainer_rejects_invalid_accumulation_steps():
    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        Trainer(
            model=ToyModel(),
            task=ToyTask(),
            optimizer=torch.optim.SGD,
            lr=1.0,
            max_epochs=1,
            accumulate_grad_batches=0,
        )


def test_trainer_accumulates_gradients_and_flushes_remainder_batches():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=CountingSGD,
        lr=1.0,
        max_epochs=1,
        accumulate_grad_batches=2,
    )

    history = trainer.fit([ToyBatch(1.0), ToyBatch(3.0), ToyBatch(5.0)])

    assert history["completed_epochs"] == 1
    assert trainer.optimizer.step_calls == 2
    assert torch.equal(trainer.model.weight.detach(), torch.tensor([6.0]))


def test_trainer_clips_gradients_before_optimizer_step():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        gradient_clip_val=1.0,
    )

    trainer.fit([ToyBatch(10.0)])

    assert torch.allclose(trainer.model.weight.detach(), torch.tensor([1.0]))


def test_trainer_steps_standard_lr_scheduler_each_epoch():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=2,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5,
        ),
    )

    trainer.fit([ToyBatch(1.0)])

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.25)


def test_trainer_rejects_invalid_lr_scheduler_interval():
    with pytest.raises(ValueError, match="lr_scheduler_interval"):
        Trainer(
            model=ToyModel(),
            task=ToyTask(),
            optimizer=torch.optim.SGD,
            lr=1.0,
            max_epochs=1,
            lr_scheduler_interval="batch",
        )


def test_trainer_steps_lr_scheduler_once_per_optimizer_step():
    callback = StepLrRecordingCallback()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        accumulate_grad_batches=2,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5,
        ),
        lr_scheduler_interval="step",
        callbacks=[callback],
    )

    trainer.fit([ToyBatch(1.0), ToyBatch(3.0), ToyBatch(5.0)])

    assert trainer.global_step == 2
    assert callback.lrs == pytest.approx([1.0, 0.5, 0.25])
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.25)


def test_sam_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="rho"):
        SAM(ToyModel().parameters(), torch.optim.SGD, lr=0.1, rho=0.0)


def test_gsam_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="alpha"):
        GSAM(ToyModel().parameters(), torch.optim.SGD, lr=0.1, alpha=-0.1)

    with pytest.raises(ValueError, match="alpha"):
        GSAM(ToyModel().parameters(), torch.optim.SGD, lr=0.1, alpha=1.1)


def test_trainer_runs_sharpness_aware_two_pass_optimizer_step():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=lambda params, lr: SAM(params, torch.optim.SGD, lr=lr, rho=0.5),
        lr=0.25,
        max_epochs=1,
    )

    history = trainer.fit([ToyBatch(2.0)])

    assert history["train"][0]["loss"] == pytest.approx(4.0)
    assert trainer.global_step == 1
    assert trainer.model.weight.detach().item() == pytest.approx(1.25)


def test_trainer_runs_adaptive_sharpness_aware_optimizer_step():
    model = ToyModel()
    with torch.no_grad():
        model.weight.fill_(2.0)

    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=lambda params, lr: ASAM(params, torch.optim.SGD, lr=lr, rho=0.5),
        lr=0.25,
        max_epochs=1,
    )

    trainer.fit([ToyBatch(4.0)])

    assert trainer.optimizer.param_groups[0]["adaptive"] is True
    assert trainer.model.weight.detach().item() == pytest.approx(3.5)


def test_trainer_runs_gsam_gradient_decomposition_step():
    trainer = Trainer(
        model=VectorToyModel([0.0, 0.0]),
        task=WeightedVectorToyTask([1.0, 10.0]),
        optimizer=lambda params, lr: GSAM(params, torch.optim.SGD, lr=lr, rho=0.5, alpha=0.2),
        lr=0.25,
        max_epochs=1,
    )

    trainer.fit([VectorToyBatch([1.0, 1.0])])

    assert trainer.global_step == 1
    assert trainer.model.weight.detach().tolist() == pytest.approx(
        [0.4951216, 7.489679],
        rel=1e-4,
        abs=1e-4,
    )


def test_trainer_rejects_sharpness_aware_optimizer_with_grad_scaler():
    with pytest.raises(ValueError, match="grad_scaler"):
        Trainer(
            model=ToyModel(),
            task=ToyTask(),
            optimizer=lambda params, lr: SAM(params, torch.optim.SGD, lr=lr),
            lr=0.25,
            max_epochs=1,
            grad_scaler=NoOpGradScaler(),
        )


def test_trainer_uses_two_forwards_for_rdrop_training_and_one_for_eval():
    batch = RDropBatch([2.0, -1.0], [0.0, 1.0], label=0)
    base_task = GraphClassificationTask(target="label", label_source="graph")
    task = RDropTask(base_task, alpha=0.5)
    model = AlternatingLogitModel()
    trainer = Trainer(
        model=model,
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=1,
    )

    history = trainer.fit([batch])

    ce_a = base_task.loss(batch, batch.logits_a, stage="train")
    ce_b = base_task.loss(batch, batch.logits_b, stage="train")
    kl_ab = F.kl_div(
        F.log_softmax(batch.logits_a, dim=-1),
        F.softmax(batch.logits_b, dim=-1),
        reduction="batchmean",
    )
    kl_ba = F.kl_div(
        F.log_softmax(batch.logits_b, dim=-1),
        F.softmax(batch.logits_a, dim=-1),
        reduction="batchmean",
    )
    expected_train_loss = (0.5 * (ce_a + ce_b) + 0.25 * (kl_ab + kl_ba)).item()

    assert model.calls == 2
    assert history["train"][0]["loss"] == pytest.approx(expected_train_loss)

    model.calls = 0
    val_summary = trainer.evaluate([batch], stage="val")

    assert model.calls == 1
    assert val_summary["loss"] == pytest.approx(base_task.loss(batch, batch.logits_a, stage="val").item())


def test_trainer_uses_monitor_value_for_plateau_scheduler(monkeypatch):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=2,
        monitor="val_loss",
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=0,
        ),
    )
    val_losses = iter([2.0, 2.0])

    def fake_run_epoch(data, stage, training):
        del data, training
        if stage == "train":
            return {"loss": 1.0}
        return {"loss": next(val_losses)}

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)

    trainer.fit([ToyBatch(1.0)], val_data=[ToyBatch(1.0)])

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_trainer_rejects_monitor_driven_scheduler_in_step_mode():
    with pytest.raises(ValueError, match="scheduler_monitor"):
        Trainer(
            model=ToyModel(),
            task=ToyTask(),
            optimizer=torch.optim.SGD,
            lr=1.0,
            max_epochs=2,
            scheduler_monitor="val_loss",
            lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=1,
                gamma=0.5,
            ),
            lr_scheduler_interval="step",
        )

    with pytest.raises(ValueError, match="ReduceLROnPlateau"):
        Trainer(
            model=ToyModel(),
            task=ToyTask(),
            optimizer=torch.optim.SGD,
            lr=1.0,
            max_epochs=2,
            lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=0,
            ),
            lr_scheduler_interval="step",
        )


def test_warmup_cosine_scheduler_rejects_invalid_configuration():
    optimizer = torch.optim.SGD(ToyModel().parameters(), lr=1.0)

    with pytest.raises(ValueError, match="warmup_epochs"):
        WarmupCosineScheduler(optimizer, warmup_epochs=0, max_epochs=5)

    with pytest.raises(ValueError, match="max_epochs"):
        WarmupCosineScheduler(optimizer, warmup_epochs=2, max_epochs=2)

    with pytest.raises(ValueError, match="min_lr_ratio"):
        WarmupCosineScheduler(optimizer, warmup_epochs=2, max_epochs=5, min_lr_ratio=-0.1)

    with pytest.raises(ValueError, match="min_lr_ratio"):
        WarmupCosineScheduler(optimizer, warmup_epochs=2, max_epochs=5, min_lr_ratio=1.1)


def test_warmup_cosine_scheduler_applies_warmup_then_cosine_decay():
    callback = LrRecordingCallback()
    trainer = Trainer(
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
        callbacks=[callback],
    )

    trainer.fit([ToyBatch(1.0)])

    assert callback.lrs == pytest.approx([0.5, 1.0, 0.775, 0.325, 0.1, 0.1])
    assert trainer.lr_scheduler.get_last_lr() == pytest.approx([0.1])


def test_layerwise_lr_decay_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="module_name_groups"):
        LayerwiseLrDecay([])

    with pytest.raises(ValueError, match="lr_decay"):
        LayerwiseLrDecay(["head"], lr_decay=0.0)

    with pytest.raises(ValueError, match="lr_decay"):
        LayerwiseLrDecay(["head"], lr_decay=1.5)


def test_trainer_uses_layerwise_lr_decay_parameter_groups():
    trainer = Trainer(
        model=FineTuneModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        optimizer_param_groups=LayerwiseLrDecay(
            ["head", "encoder_top", "encoder_bottom"],
            lr_decay=0.5,
            include_rest=False,
        ),
    )

    assert [group["lr"] for group in trainer.optimizer.param_groups] == pytest.approx([1.0, 0.5, 0.25])
    assert [group["group_name"] for group in trainer.optimizer.param_groups] == [
        "head",
        "encoder_top",
        "encoder_bottom",
    ]


def test_trainer_rejects_layerwise_lr_decay_with_unmatched_module_prefix():
    with pytest.raises(ValueError, match="matched no parameters"):
        Trainer(
            model=FineTuneModel(),
            task=ToyTask(),
            optimizer=torch.optim.SGD,
            lr=1.0,
            max_epochs=1,
            optimizer_param_groups=LayerwiseLrDecay(["missing"], include_rest=False),
        )


def test_deferred_reweighting_switches_class_weight_after_start_epoch():
    task = GraphClassificationTask(
        target="label",
        label_source="graph",
        class_count=[8.0, 2.0],
    )
    callback = DeferredReweighting(start_epoch=2, beta=0.9)
    recorder = ClassWeightRecordingCallback()
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=task,
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=3,
        callbacks=[callback, recorder],
    )

    trainer.fit([MulticlassToyBatch(0)])

    expected = _expected_drw_weight([8.0, 2.0], beta=0.9)

    assert recorder.weights[0] is None
    assert recorder.weights[1].tolist() == pytest.approx(expected.tolist())
    assert recorder.weights[2].tolist() == pytest.approx(expected.tolist())
    assert recorder.weights[3].tolist() == pytest.approx(expected.tolist())
    assert task.class_weight is None


def test_deferred_reweighting_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="start_epoch"):
        DeferredReweighting(start_epoch=0)

    with pytest.raises(ValueError, match="beta"):
        DeferredReweighting(beta=-0.1)

    with pytest.raises(ValueError, match="beta"):
        DeferredReweighting(beta=1.0)


def test_deferred_reweighting_requires_multiclass_task_with_class_count():
    trainer = Trainer(
        model=MulticlassToyModel(),
        task=GraphClassificationTask(target="label", label_source="graph"),
        optimizer=torch.optim.SGD,
        lr=0.0,
        max_epochs=1,
        callbacks=[DeferredReweighting()],
    )

    with pytest.raises(ValueError, match="class_count"):
        trainer.fit([MulticlassToyBatch(0)])
