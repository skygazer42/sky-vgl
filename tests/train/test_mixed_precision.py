from contextlib import contextmanager

import pytest
import torch
from torch import nn

from vgl.engine import AdaptiveGradientClipping, Trainer
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


class FakeScaledLoss:
    def __init__(self, loss, owner):
        self.loss = loss
        self.owner = owner

    def backward(self):
        self.owner.backward_calls += 1
        self.loss.backward()


class FakeGradScaler:
    def __init__(self):
        self.scale_calls = 0
        self.backward_calls = 0
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss):
        self.scale_calls += 1
        return FakeScaledLoss(loss, self)

    def unscale_(self, optimizer):
        del optimizer
        self.unscale_calls += 1

    def step(self, optimizer):
        self.step_calls += 1
        optimizer.step()

    def update(self):
        self.update_calls += 1


def test_trainer_rejects_unknown_precision_mode():
    with pytest.raises(ValueError, match="precision"):
        Trainer(
            model=ToyModel(),
            task=ToyTask(),
            optimizer=torch.optim.SGD,
            lr=1.0,
            max_epochs=1,
            precision="fp8-mixed",
        )


def test_trainer_rejects_fp16_mixed_with_grad_scaler_without_cuda():
    with pytest.raises(ValueError, match="fp16-mixed"):
        Trainer(
            model=ToyModel(),
            task=ToyTask(),
            optimizer=torch.optim.SGD,
            lr=1.0,
            max_epochs=1,
            precision="fp16-mixed",
            grad_scaler=FakeGradScaler(),
            device="cpu",
        )


def test_trainer_rejects_fp16_mixed_without_cuda_even_without_grad_scaler():
    with pytest.raises(ValueError, match="fp16-mixed"):
        Trainer(
            model=ToyModel(),
            task=ToyTask(),
            optimizer=torch.optim.SGD,
            lr=1.0,
            max_epochs=1,
            precision="fp16-mixed",
            device="cpu",
        )


def test_trainer_enters_autocast_for_bf16_precision(monkeypatch):
    calls = []

    @contextmanager
    def fake_autocast(device_type, dtype=None, enabled=True):
        calls.append((device_type, dtype, enabled))
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        precision="bf16-mixed",
    )

    trainer.fit([ToyBatch(1.0)])

    assert calls == [("cpu", torch.bfloat16, True)]


def test_trainer_enters_autocast_for_bf16_precision_with_explicit_cpu_device(monkeypatch):
    calls = []

    class DictBatchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

        def forward(self, batch):
            return self.weight.repeat(batch["target"].size(0))

    class DictBatchTask(Task):
        def loss(self, batch, predictions, stage):
            del stage
            return ((predictions - batch["target"]) ** 2).mean()

        def targets(self, batch, stage):
            del stage
            return batch["target"]

    @contextmanager
    def fake_autocast(device_type, dtype=None, enabled=True):
        calls.append((device_type, dtype, enabled))
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    trainer = Trainer(
        model=DictBatchModel(),
        task=DictBatchTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        precision="bf16-mixed",
        device="cpu",
    )

    trainer.fit([{"target": torch.tensor([1.0], dtype=torch.float32)}])

    assert calls == [("cpu", torch.bfloat16, True)]


def test_trainer_uses_grad_scaler_when_configured(monkeypatch):
    @contextmanager
    def fake_autocast(device_type, dtype=None, enabled=True):
        del device_type, dtype, enabled
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    scaler = FakeGradScaler()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        precision="bf16-mixed",
        grad_scaler=scaler,
        gradient_clip_val=1.0,
    )

    trainer.fit([ToyBatch(10.0)])

    assert scaler.scale_calls == 1
    assert scaler.backward_calls == 1
    assert scaler.unscale_calls == 1
    assert scaler.step_calls == 1
    assert scaler.update_calls == 1


def test_trainer_unscales_gradients_before_pre_step_callbacks(monkeypatch):
    @contextmanager
    def fake_autocast(device_type, dtype=None, enabled=True):
        del device_type, dtype, enabled
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    scaler = FakeGradScaler()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        precision="bf16-mixed",
        grad_scaler=scaler,
        callbacks=[AdaptiveGradientClipping(clipping=0.1, eps=1e-3)],
    )

    trainer.fit([ToyBatch(10.0)])

    assert scaler.unscale_calls == 1
    assert scaler.step_calls == 1
