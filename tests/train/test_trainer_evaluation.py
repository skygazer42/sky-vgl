import pytest
import torch
from torch import nn

from vgl import Graph
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


class ToyTask(Task):
    def loss(self, batch, predictions, stage):
        del stage
        return ((predictions - batch.target) ** 2).mean()

    def targets(self, batch, stage):
        del stage
        return batch.target


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
    saved_state = torch.load(checkpoint)

    assert history["best_epoch"] == 2
    assert torch.equal(trainer.model.weight.detach(), torch.tensor([0.0]))
    assert torch.equal(saved_state["weight"], torch.tensor([0.0]))


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
