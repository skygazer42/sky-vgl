from vgl.engine import TrainingHistory
from vgl.train import TrainingHistory as LegacyTrainingHistory
import pytest


def test_training_history_is_exported_from_training_packages():
    assert LegacyTrainingHistory is TrainingHistory


def test_training_history_has_legacy_train_module_alias():
    from vgl.train.history import TrainingHistory as LegacyModuleTrainingHistory

    assert LegacyModuleTrainingHistory is TrainingHistory


def test_training_history_initializes_expected_shape():
    history = TrainingHistory(epochs=3, monitor="val_loss")

    assert history == {
        "epochs": 3,
        "train": [],
        "val": [],
        "completed_epochs": 0,
        "best_epoch": None,
        "best_metric": None,
        "monitor": "val_loss",
        "stopped_early": False,
        "stop_reason": None,
        "fit_elapsed_seconds": None,
        "epoch_elapsed_seconds": [],
        "final_train": None,
        "final_val": None,
        "run_name": None,
        "root_dir": None,
        "fast_dev_run": False,
        "sanity_check_passed": False,
        "profiler": None,
        "profile": None,
    }


def test_training_history_records_epochs_and_stop_state():
    history = TrainingHistory(epochs=5, monitor="train_loss")

    history.record_epoch(
        epoch=1,
        train_summary={"loss": 1.0},
        val_summary=None,
        best_epoch=1,
        best_metric=1.0,
    )
    history.record_epoch(
        epoch=2,
        train_summary={"loss": 0.5},
        val_summary={"loss": 0.75},
        best_epoch=2,
        best_metric=0.75,
    )
    history.mark_stopped("requested stop")

    assert history["train"] == [{"loss": 1.0}, {"loss": 0.5}]
    assert history["val"] == [{"loss": 0.75}]
    assert history["completed_epochs"] == 2
    assert history["best_epoch"] == 2
    assert history["best_metric"] == 0.75
    assert history["stopped_early"] is True
    assert history["stop_reason"] == "requested stop"
    assert history["epoch_elapsed_seconds"] == [None, None]
    assert history["final_train"] is None
    assert history["final_val"] is None


def test_training_history_from_state_dict_rejects_missing_required_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        TrainingHistory.from_state_dict({"epochs": 3})


def test_training_history_from_state_dict_normalizes_simple_profiler_profile_schema():
    history = TrainingHistory.from_state_dict(
        {
            "epochs": 3,
            "monitor": "val_loss",
            "profiler": "simple",
            "profile": {
                "forward_seconds_total": 1.5,
                "train_step_count": 3,
                "unknown_extra": 99.0,
            },
        }
    )

    assert tuple(history["profile"]) == (
        "batch_materialization_seconds_total",
        "forward_seconds_total",
        "backward_seconds_total",
        "optimizer_step_seconds_total",
        "train_step_seconds_total",
        "train_stage_seconds_total",
        "val_stage_seconds_total",
        "test_stage_seconds_total",
        "sanity_val_stage_seconds_total",
        "train_step_count",
        "train_step_seconds_avg",
    )
    assert history["profile"]["forward_seconds_total"] == 1.5
    assert history["profile"]["train_step_count"] == 3
    assert history["profile"]["train_step_seconds_avg"] == 0.0
    assert "unknown_extra" not in history["profile"]


def test_training_history_from_state_dict_rejects_non_mapping_profile():
    with pytest.raises(ValueError, match="profile must be a mapping"):
        TrainingHistory.from_state_dict(
            {
                "epochs": 3,
                "monitor": "val_loss",
                "profiler": "simple",
                "profile": ["bad"],
            }
        )


def test_training_history_from_state_dict_rejects_completed_epochs_past_total_epochs():
    with pytest.raises(ValueError, match="completed_epochs"):
        TrainingHistory.from_state_dict(
            {
                "epochs": 3,
                "monitor": "val_loss",
                "completed_epochs": 4,
            }
        )


def test_training_history_from_state_dict_rejects_best_epoch_past_completed_epochs():
    with pytest.raises(ValueError, match="best_epoch"):
        TrainingHistory.from_state_dict(
            {
                "epochs": 3,
                "monitor": "val_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
                "best_epoch": 3,
            }
        )


def test_training_history_from_state_dict_rejects_best_metric_without_best_epoch():
    with pytest.raises(ValueError, match="best_metric"):
        TrainingHistory.from_state_dict(
            {
                "epochs": 3,
                "monitor": "val_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1, 0.2],
                "best_metric": 0.75,
            }
        )


def test_training_history_from_state_dict_rejects_train_history_length_mismatch():
    with pytest.raises(ValueError, match="train history length"):
        TrainingHistory.from_state_dict(
            {
                "epochs": 3,
                "monitor": "val_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}],
            }
        )


def test_training_history_from_state_dict_rejects_epoch_elapsed_length_mismatch():
    with pytest.raises(ValueError, match="epoch_elapsed_seconds"):
        TrainingHistory.from_state_dict(
            {
                "epochs": 3,
                "monitor": "val_loss",
                "completed_epochs": 2,
                "train": [{"loss": 1.0}, {"loss": 0.5}],
                "epoch_elapsed_seconds": [0.1],
            }
        )
