import pytest
import torch
from torch import nn

from vgl import Graph
from vgl.engine import trainer as trainer_module
from vgl.train.tasks import NodeClassificationTask
from vgl.train.trainer import Trainer


class LinearNodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.x)


def test_trainer_runs_single_epoch():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
        val_mask=torch.tensor([True, True]),
        test_mask=torch.tensor([True, True]),
    )
    model = LinearNodeModel()
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
    )
    trainer = Trainer(
        model=model,
        task=task,
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(graph)

    assert history["epochs"] == 1
    assert len(history["train"]) == 1


def test_validate_init_config_accepts_supported_values():
    trainer_module._validate_init_config(
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        gradient_clip_val=0.0,
        console_flush_every_n_steps=1,
        scheduler_monitor=None,
        lr_scheduler=object(),
        lr_scheduler_interval="epoch",
        precision="32",
        non_blocking=True,
        num_sanity_val_steps=0,
        profiler="simple",
    )


def test_validate_init_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="scheduler_monitor requires lr_scheduler"):
        trainer_module._validate_init_config(
            accumulate_grad_batches=1,
            log_every_n_steps=1,
            gradient_clip_val=None,
            console_flush_every_n_steps=None,
            scheduler_monitor="val_loss",
            lr_scheduler=None,
            lr_scheduler_interval="epoch",
            precision="32",
            non_blocking=None,
            num_sanity_val_steps=0,
            profiler=None,
        )

    with pytest.raises(ValueError, match="precision must be one of"):
        trainer_module._validate_init_config(
            accumulate_grad_batches=1,
            log_every_n_steps=1,
            gradient_clip_val=None,
            console_flush_every_n_steps=None,
            scheduler_monitor=None,
            lr_scheduler=object(),
            lr_scheduler_interval="epoch",
            precision="fp32",
            non_blocking=None,
            num_sanity_val_steps=0,
            profiler=None,
        )

    with pytest.raises(TypeError, match="non_blocking must be None or a bool"):
        trainer_module._validate_init_config(
            accumulate_grad_batches=1,
            log_every_n_steps=1,
            gradient_clip_val=None,
            console_flush_every_n_steps=None,
            scheduler_monitor=None,
            lr_scheduler=object(),
            lr_scheduler_interval="epoch",
            precision="32",
            non_blocking="yes",
            num_sanity_val_steps=0,
            profiler=None,
        )


def test_trainer_rejects_non_blocking_when_batch_device_transfer_is_disabled():
    with pytest.raises(ValueError, match="non_blocking"):
        Trainer(
            model=LinearNodeModel(),
            task=NodeClassificationTask(
                target="y",
                split=("train_mask", "val_mask", "test_mask"),
            ),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=1,
            move_batch_to_device=False,
            non_blocking=True,
        )
