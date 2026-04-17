import pytest
import torch
from torch import nn
from pathlib import Path

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


def test_normalize_init_config_returns_normalized_settings():
    config = trainer_module._normalize_init_config(
        accumulate_grad_batches=2,
        log_every_n_steps=5,
        gradient_clip_val=0,
        console_flush_every_n_steps=3,
        scheduler_monitor=None,
        lr_scheduler=object(),
        lr_scheduler_interval="epoch",
        precision="32",
        device="cpu",
        move_batch_to_device=True,
        non_blocking=False,
        default_root_dir="artifacts",
        run_name="smoke",
        fast_dev_run=True,
        limit_train_batches=1.0,
        limit_val_batches=2,
        limit_test_batches=3,
        val_check_interval=1,
        num_sanity_val_steps=0,
        profiler="simple",
    )

    assert config.device == torch.device("cpu")
    assert config.default_root_dir == Path("artifacts")
    assert config.run_name == "smoke"
    assert config.fast_dev_run is True
    assert config.fast_dev_run_batches == 1
    assert config.limit_train_batches == 1.0
    assert config.limit_val_batches == 2
    assert config.limit_test_batches == 3
    assert config.val_check_interval == 1
    assert config.accumulate_grad_batches == 2
    assert config.log_every_n_steps == 5
    assert config.gradient_clip_val == 0.0
    assert config.console_flush_every_n_steps == 3


def test_normalize_init_config_rejects_non_blocking_without_device_transfer():
    with pytest.raises(ValueError, match="non_blocking requires move_batch_to_device=True"):
        trainer_module._normalize_init_config(
            accumulate_grad_batches=1,
            log_every_n_steps=1,
            gradient_clip_val=None,
            console_flush_every_n_steps=None,
            scheduler_monitor=None,
            lr_scheduler=None,
            lr_scheduler_interval="epoch",
            precision="32",
            device=None,
            move_batch_to_device=False,
            non_blocking=True,
            default_root_dir=None,
            run_name=None,
            fast_dev_run=False,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            val_check_interval=1.0,
            num_sanity_val_steps=0,
            profiler=None,
        )


def test_build_init_config_normalizes_constructor_values(tmp_path):
    config = trainer_module._build_init_config(
        model=LinearNodeModel(),
        device="cpu",
        move_batch_to_device=1,
        non_blocking=None,
        default_root_dir=tmp_path / "artifacts",
        run_name=123,
        fast_dev_run=True,
        limit_train_batches=2,
        limit_val_batches=0.5,
        limit_test_batches=3,
        val_check_interval=1,
        num_sanity_val_steps=2,
        profiler="simple",
        accumulate_grad_batches=2,
        log_every_n_steps=5,
        gradient_clip_val=1,
        console_flush_every_n_steps=10,
        scheduler_monitor=None,
        lr_scheduler=None,
        lr_scheduler_interval="epoch",
        precision="32",
    )

    assert config["device"] == torch.device("cpu")
    assert config["move_batch_to_device"] is True
    assert config["default_root_dir"] == tmp_path / "artifacts"
    assert config["run_name"] == "123"
    assert config["fast_dev_run_batches"] == 1
    assert config["limit_train_batches"] == 2
    assert config["limit_val_batches"] == 0.5
    assert config["limit_test_batches"] == 3
    assert config["val_check_interval"] == 1
    assert config["num_sanity_val_steps"] == 2
    assert config["accumulate_grad_batches"] == 2
    assert config["log_every_n_steps"] == 5
    assert config["gradient_clip_val"] == 1.0


def test_build_init_config_rejects_non_blocking_without_transfer():
    with pytest.raises(ValueError, match="non_blocking requires move_batch_to_device=True"):
        trainer_module._build_init_config(
            model=LinearNodeModel(),
            device=None,
            move_batch_to_device=False,
            non_blocking=True,
            default_root_dir=None,
            run_name=None,
            fast_dev_run=False,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            val_check_interval=1.0,
            num_sanity_val_steps=0,
            profiler=None,
            accumulate_grad_batches=1,
            log_every_n_steps=1,
            gradient_clip_val=None,
            console_flush_every_n_steps=None,
            scheduler_monitor=None,
            lr_scheduler=None,
            lr_scheduler_interval="epoch",
            precision="32",
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
