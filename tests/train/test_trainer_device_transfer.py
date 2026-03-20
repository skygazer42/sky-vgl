import pytest
import torch
from torch import nn

from vgl import Graph
from vgl.core.batch import NodeBatch
from vgl.data.sample import SampleRecord
from vgl.engine import Trainer
from vgl.train.task import Task


class DictBatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))
        self.last_target_device_type = None
        self.last_aux_device_type = None

    def forward(self, batch):
        self.last_target_device_type = batch["target"].device.type
        self.last_aux_device_type = batch["nested"][0].device.type
        return self.weight.repeat(batch["target"].size(0))


class DictBatchTask(Task):
    def loss(self, batch, predictions, stage):
        del stage
        return ((predictions - batch["target"]) ** 2).mean()

    def targets(self, batch, stage):
        del stage
        return batch["target"]


class NodeBatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))
        self.seed_index_device_type = None
        self.node_feature_device_type = None
        self.edge_index_device_type = None

    def forward(self, batch):
        self.seed_index_device_type = batch.seed_index.device.type
        self.node_feature_device_type = batch.graph.x.device.type
        self.edge_index_device_type = batch.graph.edge_index.device.type
        return self.weight.repeat(batch.seed_index.size(0))


class NodeBatchTask(Task):
    def loss(self, batch, predictions, stage):
        del stage
        return (predictions**2).mean()

    def targets(self, batch, stage):
        del stage
        return torch.zeros(batch.seed_index.size(0), dtype=torch.float32, device=batch.seed_index.device)


class ToRaisingBatch:
    def __init__(self):
        self.target = torch.tensor([1.0], dtype=torch.float32)

    def to(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("automatic batch transfer should be disabled")


class AttrBatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, batch):
        return self.weight.repeat(batch.target.size(0))


class AttrBatchTask(Task):
    def loss(self, batch, predictions, stage):
        del stage
        return ((predictions - batch.target) ** 2).mean()

    def targets(self, batch, stage):
        del stage
        return batch.target


class UnsupportedBatch:
    def __init__(self):
        self.target = torch.tensor([1.0], dtype=torch.float32)


def _node_batch():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    sample = SampleRecord(
        graph=graph,
        metadata={"seed": 1, "sample_id": "a"},
        sample_id="a",
        subgraph_seed=1,
    )
    return NodeBatch.from_samples([sample])


def test_trainer_device_moves_model_and_standard_container_batches():
    model = DictBatchModel()
    trainer = Trainer(
        model=model,
        task=DictBatchTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        device="cpu",
    )

    trainer.fit(
        [
            {
                "target": torch.tensor([1.0], dtype=torch.float32),
                "nested": [torch.tensor([2.0], dtype=torch.float32)],
            }
        ]
    )

    assert next(model.parameters()).device.type == "cpu"
    assert model.last_target_device_type == "cpu"
    assert model.last_aux_device_type == "cpu"


def test_trainer_device_moves_node_batch_tensors_before_forward():
    model = NodeBatchModel()
    trainer = Trainer(
        model=model,
        task=NodeBatchTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        device="cpu",
    )

    trainer.fit([_node_batch()])

    assert model.seed_index_device_type == "cpu"
    assert model.node_feature_device_type == "cpu"
    assert model.edge_index_device_type == "cpu"


def test_trainer_can_skip_batch_auto_movement_while_moving_model():
    model = AttrBatchModel()
    trainer = Trainer(
        model=model,
        task=AttrBatchTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        device="cpu",
        move_batch_to_device=False,
    )

    trainer.fit([ToRaisingBatch()])

    assert next(model.parameters()).device.type == "cpu"


def test_trainer_rejects_unsupported_batch_type_when_auto_move_enabled():
    trainer = Trainer(
        model=AttrBatchModel(),
        task=AttrBatchTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        device="cpu",
    )

    with pytest.raises(TypeError, match="Unsupported batch type"):
        trainer.fit([UnsupportedBatch()])
