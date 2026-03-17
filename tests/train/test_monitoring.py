import pytest

from vgl.engine.monitoring import extract_monitor_value, is_improvement, resolve_monitor


def test_resolve_monitor_defaults_to_val_loss_when_validation_data_exists():
    assert resolve_monitor(None, has_val_data=True) == ("val_loss", "min")


def test_resolve_monitor_defaults_to_train_loss_without_validation_data():
    assert resolve_monitor(None, has_val_data=False) == ("train_loss", "min")


def test_resolve_monitor_rejects_val_monitor_without_validation_data():
    with pytest.raises(ValueError, match="val_data"):
        resolve_monitor("val_accuracy", has_val_data=False)


def test_extract_monitor_value_reads_requested_stage_metric():
    value = extract_monitor_value(
        "val_accuracy",
        train_summary={"loss": 1.0},
        val_summary={"accuracy": 0.75},
    )

    assert value == 0.75


def test_is_improvement_respects_mode_and_min_delta():
    assert is_improvement(0.90, 1.00, "min", min_delta=0.05) is True
    assert is_improvement(0.97, 1.00, "min", min_delta=0.05) is False
    assert is_improvement(0.80, None, "max", min_delta=0.10) is True
