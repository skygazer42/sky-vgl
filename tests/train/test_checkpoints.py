import torch
from torch import nn
import pytest

from vgl._artifact import ARTIFACT_FORMAT_KEY, ARTIFACT_FORMAT_VERSION_KEY
from vgl.engine.checkpoints import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    LEGACY_CHECKPOINT_FORMAT,
    LEGACY_CHECKPOINT_FORMAT_VERSION,
    checkpoint_event_fields,
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
        ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
        ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
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
        ARTIFACT_FORMAT_KEY: LEGACY_CHECKPOINT_FORMAT,
        ARTIFACT_FORMAT_VERSION_KEY: LEGACY_CHECKPOINT_FORMAT_VERSION,
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
        callback_states=[
            {"callback": "pkg.CallbackA", "index": 0, "state": {"records": [1]}},
            {"callback": "pkg.CallbackB", "index": 1, "state": {"shadow": 2}},
        ],
        trainer_state={
            "global_step": 3,
            "best_epoch": 2,
            "best_metric": 1.0,
            "best_state_dict": {"weight": torch.tensor([3.0])},
        },
        history_state={
            "epochs": 5,
            "monitor": "val_loss",
            "completed_epochs": 2,
            "train": [{"loss": 1.0}, {"loss": 0.5}],
            "epoch_elapsed_seconds": [0.1, 0.2],
            "best_epoch": 2,
            "best_metric": 1.0,
        },
    )
    payload = load_checkpoint(checkpoint)

    assert payload == {
        ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
        ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": {"weight": torch.tensor([4.0])},
        "metadata": {"epoch": 5},
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 0.1}]},
        "lr_scheduler_state_dict": {"last_epoch": 2},
        "grad_scaler_state_dict": {"scale": 1024.0},
        "callback_states": [
            {"callback": "pkg.CallbackA", "index": 0, "state": {"records": [1]}},
            {"callback": "pkg.CallbackB", "index": 1, "state": {"shadow": 2}},
        ],
        "trainer_state": {
            "global_step": 3,
            "best_epoch": 2,
            "best_metric": 1.0,
            "best_state_dict": {"weight": torch.tensor([3.0])},
        },
        "history_state": {
            "epochs": 5,
            "monitor": "val_loss",
            "completed_epochs": 2,
            "train": [{"loss": 1.0}, {"loss": 0.5}],
            "epoch_elapsed_seconds": [0.1, 0.2],
            "best_epoch": 2,
            "best_metric": 1.0,
            "val": [],
            "stopped_early": False,
            "stop_reason": None,
            "fit_elapsed_seconds": None,
            "final_train": None,
            "final_val": None,
            "run_name": None,
            "root_dir": None,
            "fast_dev_run": False,
            "sanity_check_passed": False,
            "profiler": None,
            "profile": None,
        },
    }


def test_load_checkpoint_normalizes_training_state_sections(tmp_path):
    checkpoint = tmp_path / "normalized-sections.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([4.0])},
            "metadata": {"epoch": 5},
            "trainer_state": {
                "best_epoch": "2",
                "best_metric": "1.0",
                "best_state_dict": {"weight": torch.tensor([3.0])},
                "global_step": "3",
                "fit_elapsed_seconds": "1.5",
                "fit_profile": {"forward_seconds_total": "2.0", "train_step_count": "4"},
            },
            "history_state": {
                "epochs": "5",
                "monitor": "val_loss",
                "completed_epochs": "2",
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": ["0.1", "0.2"],
                "best_epoch": "2",
                "best_metric": "1.0",
                "run_name": 123,
                "root_dir": 456,
                "fast_dev_run": 1,
            },
        },
        checkpoint,
    )

    payload = load_checkpoint(checkpoint)

    assert payload["history_state"]["epochs"] == 5
    assert payload["history_state"]["completed_epochs"] == 2
    assert payload["history_state"]["best_epoch"] == 2
    assert payload["history_state"]["best_metric"] == 1.0
    assert payload["history_state"]["run_name"] == "123"
    assert payload["history_state"]["root_dir"] == "456"
    assert payload["history_state"]["fast_dev_run"] is True
    assert payload["trainer_state"]["best_epoch"] == "2"
    assert payload["trainer_state"]["best_metric"] == "1.0"
    assert payload["trainer_state"]["global_step"] == "3"
    assert payload["trainer_state"]["fit_elapsed_seconds"] == "1.5"
    assert payload["trainer_state"]["fit_profile"]["forward_seconds_total"] == "2.0"
    assert payload["trainer_state"]["fit_profile"]["train_step_count"] == "4"


def test_checkpoint_event_fields_include_checkpoint_artifact_metadata(tmp_path):
    checkpoint = tmp_path / "event.pt"
    save_checkpoint(checkpoint, {"weight": torch.tensor([1.0])})

    event_fields = checkpoint_event_fields(checkpoint, save_seconds=0.25)

    assert event_fields[ARTIFACT_FORMAT_KEY] == CHECKPOINT_FORMAT
    assert event_fields[ARTIFACT_FORMAT_VERSION_KEY] == CHECKPOINT_FORMAT_VERSION
    assert event_fields["path"] == str(checkpoint)
    assert event_fields["size_bytes"] > 0
    assert event_fields["save_seconds"] == 0.25


def test_load_checkpoint_rejects_non_mapping_history_state(tmp_path):
    checkpoint = tmp_path / "bad-history.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "metadata": {},
            "history_state": ["not", "a", "mapping"],
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="history_state must be a mapping"):
        load_checkpoint(checkpoint)


def test_save_checkpoint_rejects_non_mapping_metadata_even_when_falsy(tmp_path):
    checkpoint = tmp_path / "bad-save-metadata.pt"

    with pytest.raises(ValueError, match="metadata must be a mapping"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            metadata=[],
        )


def test_save_checkpoint_rejects_non_mapping_model_state_dict(tmp_path):
    checkpoint = tmp_path / "bad-save-model-state.pt"

    with pytest.raises(ValueError, match="model_state_dict must be a mapping"):
        save_checkpoint(
            checkpoint,
            ["bad"],
        )


def test_save_checkpoint_rejects_malformed_callback_states(tmp_path):
    checkpoint = tmp_path / "bad-save-callbacks.pt"

    with pytest.raises(ValueError, match="callback_states entry callback must be a string"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            callback_states=[{"state": {}}],
        )


def test_save_checkpoint_rejects_trainer_state_best_epoch_without_best_metric(tmp_path):
    checkpoint = tmp_path / "bad-save-trainer-state-best-metric.pt"

    with pytest.raises(ValueError, match="best_metric"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={"best_epoch": 1},
        )


def test_save_checkpoint_rejects_non_string_trainer_state_active_monitor(tmp_path):
    checkpoint = tmp_path / "bad-save-trainer-state-monitor.pt"

    with pytest.raises(ValueError, match="active_monitor"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={"active_monitor": ["bad"]},
        )


def test_save_checkpoint_rejects_missing_global_step_when_history_has_progress(tmp_path):
    checkpoint = tmp_path / "bad-save-global-step-missing.pt"

    with pytest.raises(ValueError, match="global_step"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={},
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
            },
        )


def test_save_checkpoint_rejects_missing_trainer_state_when_history_has_progress(tmp_path):
    checkpoint = tmp_path / "bad-save-trainer-state-missing.pt"

    with pytest.raises(ValueError, match="global_step"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
            },
        )


def test_save_checkpoint_rejects_global_step_behind_history_progress(tmp_path):
    checkpoint = tmp_path / "bad-save-global-step-range.pt"

    with pytest.raises(ValueError, match="global_step"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={"global_step": 1},
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
            },
        )


def test_save_checkpoint_rejects_best_epoch_mismatch_with_history_state(tmp_path):
    checkpoint = tmp_path / "bad-save-best-epoch-mismatch.pt"

    with pytest.raises(ValueError, match="best_epoch"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={
                "global_step": 2,
                "best_epoch": 1,
                "best_metric": 1.0,
                "best_state_dict": {"weight": torch.tensor([1.0])},
            },
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
                "best_epoch": 2,
                "best_metric": 1.0,
            },
        )


def test_save_checkpoint_rejects_best_metric_mismatch_with_history_state(tmp_path):
    checkpoint = tmp_path / "bad-save-best-metric-mismatch.pt"

    with pytest.raises(ValueError, match="best_metric"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={
                "global_step": 2,
                "best_epoch": 2,
                "best_metric": 1.0,
                "best_state_dict": {"weight": torch.tensor([1.0])},
            },
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
                "best_epoch": 2,
                "best_metric": 2.0,
            },
        )


def test_save_checkpoint_rejects_missing_trainer_best_epoch_when_history_has_one(tmp_path):
    checkpoint = tmp_path / "bad-save-missing-trainer-best-epoch.pt"

    with pytest.raises(ValueError, match="best_epoch"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={"global_step": 2},
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
                "best_epoch": 2,
                "best_metric": 1.0,
            },
        )


def test_save_checkpoint_rejects_missing_trainer_best_metric_when_history_has_one(tmp_path):
    checkpoint = tmp_path / "bad-save-missing-trainer-best-metric.pt"

    with pytest.raises(ValueError, match="best_metric"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={"global_step": 2, "best_epoch": 2, "best_state_dict": {"weight": torch.tensor([1.0])}},
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
                "best_epoch": 2,
                "best_metric": 1.0,
            },
        )


def test_save_checkpoint_rejects_missing_history_best_metric_when_trainer_has_one(tmp_path):
    checkpoint = tmp_path / "bad-save-missing-history-best-metric.pt"

    with pytest.raises(ValueError, match="best_metric"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={
                "global_step": 2,
                "best_epoch": 2,
                "best_metric": 1.0,
                "best_state_dict": {"weight": torch.tensor([1.0])},
            },
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
                "best_epoch": 2,
            },
        )


def test_save_checkpoint_rejects_missing_history_best_metric_when_trainer_has_best_state(tmp_path):
    checkpoint = tmp_path / "bad-save-missing-history-best-metric-2.pt"

    with pytest.raises(ValueError, match="best_metric"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={
                "global_step": 2,
                "best_epoch": 2,
                "best_metric": 1.0,
                "best_state_dict": {"weight": torch.tensor([1.0])},
            },
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
                "best_epoch": 2,
            },
        )


def test_save_checkpoint_rejects_active_monitor_mismatch_with_history_state(tmp_path):
    checkpoint = tmp_path / "bad-save-active-monitor-mismatch.pt"

    with pytest.raises(ValueError, match="active_monitor"):
        save_checkpoint(
            checkpoint,
            {"weight": torch.tensor([1.0])},
            trainer_state={"active_monitor": "val_loss"},
            history_state={
                "epochs": 4,
                "monitor": "train_loss",
                "completed_epochs": 0,
            },
        )


def test_load_checkpoint_rejects_non_mapping_metadata_even_when_falsy(tmp_path):
    checkpoint = tmp_path / "bad-metadata.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "metadata": [],
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="metadata must be a mapping"):
        load_checkpoint(checkpoint)


def test_load_checkpoint_rejects_non_mapping_model_state_dict(tmp_path):
    checkpoint = tmp_path / "bad-model-state.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": ["not", "a", "mapping"],
            "metadata": {},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="model_state_dict must be a mapping"):
        load_checkpoint(checkpoint)


def test_load_checkpoint_rejects_non_sequence_callback_states(tmp_path):
    checkpoint = tmp_path / "bad-callbacks.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "metadata": {},
            "callback_states": {"callback": "x"},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="callback_states must be a sequence"):
        load_checkpoint(checkpoint)


def test_load_checkpoint_rejects_non_mapping_callback_state_entries(tmp_path):
    checkpoint = tmp_path / "bad-callback-entry.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "metadata": {},
            "callback_states": ["bad-entry"],
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="callback_states entries must be mappings"):
        load_checkpoint(checkpoint)


def test_load_checkpoint_rejects_non_string_callback_state_name(tmp_path):
    checkpoint = tmp_path / "bad-callback-name.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "metadata": {},
            "callback_states": [{"callback": 123, "index": 0, "state": {}}],
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="callback_states entry callback must be a string"):
        load_checkpoint(checkpoint)


def test_load_checkpoint_rejects_non_integer_callback_state_index(tmp_path):
    checkpoint = tmp_path / "bad-callback-index.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "metadata": {},
            "callback_states": [{"callback": "pkg.Callback", "index": "bad", "state": {}}],
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="callback_states entry index must be an integer"):
        load_checkpoint(checkpoint)


def test_load_checkpoint_rejects_negative_callback_state_index(tmp_path):
    checkpoint = tmp_path / "bad-callback-negative-index.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "metadata": {},
            "callback_states": [{"callback": "pkg.Callback", "index": -1, "state": {}}],
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="callback_states entry index must be >= 0"):
        load_checkpoint(checkpoint)


def test_load_checkpoint_rejects_non_mapping_callback_state_payload(tmp_path):
    checkpoint = tmp_path / "bad-callback-state-payload.pt"
    torch.save(
        {
            ARTIFACT_FORMAT_KEY: CHECKPOINT_FORMAT,
            ARTIFACT_FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "metadata": {},
            "callback_states": [{"callback": "pkg.Callback", "index": 0, "state": ["bad"]}],
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="callback_states entry state must be a mapping"):
        load_checkpoint(checkpoint)
