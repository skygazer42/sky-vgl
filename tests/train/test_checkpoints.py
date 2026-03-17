import torch
from torch import nn

from vgl.engine.checkpoints import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))


def test_legacy_train_package_reexports_checkpoint_helpers():
    from vgl.engine import (
        CHECKPOINT_FORMAT as EngineCheckpointFormat,
        CHECKPOINT_FORMAT_VERSION as EngineCheckpointFormatVersion,
        load_checkpoint as engine_load_checkpoint,
        restore_checkpoint as engine_restore_checkpoint,
        save_checkpoint as engine_save_checkpoint,
    )
    from vgl.train import (
        CHECKPOINT_FORMAT as LegacyCheckpointFormat,
        CHECKPOINT_FORMAT_VERSION as LegacyCheckpointFormatVersion,
        load_checkpoint as legacy_load_checkpoint,
        restore_checkpoint as legacy_restore_checkpoint,
        save_checkpoint as legacy_save_checkpoint,
    )
    from vgl.train.checkpoints import (
        CHECKPOINT_FORMAT as LegacyModuleCheckpointFormat,
        CHECKPOINT_FORMAT_VERSION as LegacyModuleCheckpointFormatVersion,
        load_checkpoint as legacy_module_load_checkpoint,
        restore_checkpoint as legacy_module_restore_checkpoint,
        save_checkpoint as legacy_module_save_checkpoint,
    )

    assert LegacyCheckpointFormat == EngineCheckpointFormat
    assert LegacyCheckpointFormatVersion == EngineCheckpointFormatVersion
    assert legacy_load_checkpoint is engine_load_checkpoint
    assert legacy_restore_checkpoint is engine_restore_checkpoint
    assert legacy_save_checkpoint is engine_save_checkpoint
    assert LegacyModuleCheckpointFormat == EngineCheckpointFormat
    assert LegacyModuleCheckpointFormatVersion == EngineCheckpointFormatVersion
    assert legacy_module_load_checkpoint is engine_load_checkpoint
    assert legacy_module_restore_checkpoint is engine_restore_checkpoint
    assert legacy_module_save_checkpoint is engine_save_checkpoint


def test_save_and_load_checkpoint_round_trip(tmp_path):
    checkpoint = tmp_path / "round-trip.pt"

    save_checkpoint(checkpoint, {"weight": torch.tensor([4.0])}, metadata={"epoch": 5})
    payload = load_checkpoint(checkpoint)

    assert payload == {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": {"weight": torch.tensor([4.0])},
        "metadata": {"epoch": 5},
    }


def test_restore_checkpoint_loads_legacy_state_dict_into_model(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    model = ToyModel()
    torch.save({"weight": torch.tensor([6.0])}, checkpoint)

    payload = restore_checkpoint(model, checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([6.0]))
    assert payload == {
        "format": "legacy.state_dict",
        "format_version": 0,
        "model_state_dict": {"weight": torch.tensor([6.0])},
        "metadata": {},
    }


def test_save_and_load_checkpoint_round_trip_with_training_state_sections(tmp_path):
    checkpoint = tmp_path / "training.pt"

    save_checkpoint(
        checkpoint,
        {"weight": torch.tensor([4.0])},
        metadata={"epoch": 5},
        optimizer_state_dict={"state": {}, "param_groups": [{"lr": 0.1}]},
        lr_scheduler_state_dict={"last_epoch": 2},
        grad_scaler_state_dict={"scale": 1024.0},
        callback_states=[{"records": [1]}, {"shadow": 2}],
        trainer_state={"global_step": 3, "best_epoch": 2},
        history_state={"epochs": 5, "completed_epochs": 2},
    )
    payload = load_checkpoint(checkpoint)

    assert payload == {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": {"weight": torch.tensor([4.0])},
        "metadata": {"epoch": 5},
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 0.1}]},
        "lr_scheduler_state_dict": {"last_epoch": 2},
        "grad_scaler_state_dict": {"scale": 1024.0},
        "callback_states": [{"records": [1]}, {"shadow": 2}],
        "trainer_state": {"global_step": 3, "best_epoch": 2},
        "history_state": {"epochs": 5, "completed_epochs": 2},
    }
