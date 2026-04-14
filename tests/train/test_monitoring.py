import pytest

from vgl.engine.monitoring import extract_monitor_value, is_improvement, resolve_monitor, resolve_monitor_mode


def test_resolve_monitor_defaults_to_val_loss_when_validation_data_exists():
    assert resolve_monitor(None, has_val_data=True) == ("val_loss", "min")


def test_resolve_monitor_defaults_to_train_loss_without_validation_data():
    assert resolve_monitor(None, has_val_data=False) == ("train_loss", "min")


def test_resolve_monitor_rejects_val_monitor_without_validation_data():
    with pytest.raises(ValueError, match="val_data"):
        resolve_monitor("val_accuracy", has_val_data=False)


def test_resolve_monitor_rejects_stage_less_or_malformed_keys():
    with pytest.raises(ValueError, match="train_<metric>' or 'val_<metric>"):
        resolve_monitor("accuracy", has_val_data=True)

    with pytest.raises(ValueError, match="train_<metric>' or 'val_<metric>"):
        resolve_monitor("train", has_val_data=False)


def test_resolve_monitor_mode_defaults_loss_like_metrics_to_min():
    assert resolve_monitor_mode("val_loss") == "min"
    assert resolve_monitor_mode("val_total_loss") == "min"
    assert resolve_monitor_mode("train_recon_loss") == "min"
    assert resolve_monitor_mode("val_accuracy") == "max"


def test_extract_monitor_value_reads_requested_stage_metric():
    value = extract_monitor_value(
        "val_accuracy",
        train_summary={"loss": 1.0},
        val_summary={"accuracy": 0.75},
    )

    assert value == 0.75


def test_extract_monitor_value_rejects_unknown_or_malformed_keys():
    with pytest.raises(ValueError, match="train_<metric>' or 'val_<metric>"):
        extract_monitor_value(
            "loss",
            train_summary={"loss": 1.0},
            val_summary={"loss": 0.5},
        )

    with pytest.raises(ValueError, match="was not produced"):
        extract_monitor_value(
            "val_accuracy",
            train_summary={"loss": 1.0},
            val_summary={"loss": 0.5},
        )


def test_is_improvement_respects_mode_and_min_delta():
    assert is_improvement(0.90, 1.00, "min", min_delta=0.05) is True
    assert is_improvement(0.97, 1.00, "min", min_delta=0.05) is False
    assert is_improvement(0.80, None, "max", min_delta=0.10) is True
