_MONITOR_STAGES = {"train", "val"}
_SUMMARY_STAGES = ("train", "val", "test")
_RESERVED_STAGE_PREFIXES = tuple(f"{stage}_" for stage in _SUMMARY_STAGES)


def _split_monitor_key(monitor):
    stage, separator, key = str(monitor).partition("_")
    if separator == "" or stage not in _MONITOR_STAGES or key == "":
        raise ValueError("Trainer monitor must use 'train_<metric>' or 'val_<metric>'")
    return stage, key


def _default_monitor_mode_for_key(key):
    if key == "loss" or key.endswith("_loss"):
        return "min"
    return "max"


def validate_metric_name(name):
    resolved = str(name)
    if any(resolved.startswith(prefix) for prefix in _RESERVED_STAGE_PREFIXES):
        raise ValueError(
            "Metric names must not start with reserved stage prefixes: train_, val_, test_"
        )
    return resolved


def resolve_monitor_mode(monitor, mode=None, *, invalid_mode_message="monitor_mode must be 'min' or 'max'"):
    _, key = _split_monitor_key(monitor)
    resolved_mode = mode or _default_monitor_mode_for_key(key)
    if resolved_mode not in {"min", "max"}:
        raise ValueError(invalid_mode_message)
    return resolved_mode


def resolve_monitor(monitor, *, has_val_data, mode=None):
    resolved_monitor = monitor or ("val_loss" if has_val_data else "train_loss")
    stage, _ = _split_monitor_key(resolved_monitor)
    if stage == "val" and not has_val_data:
        raise ValueError("Trainer monitor requires val_data for val_* keys")
    return resolved_monitor, resolve_monitor_mode(resolved_monitor, mode=mode)


def extract_monitor_value(
    monitor,
    *,
    train_summary,
    val_summary,
    error_subject="Monitor key",
):
    stage, key = _split_monitor_key(monitor)
    source = {"train": train_summary, "val": val_summary}.get(stage)
    if source is None or key not in source:
        raise ValueError(f"{error_subject} {monitor} was not produced by the trainer")
    return float(source[key])


def is_improvement(current, best, mode, *, min_delta=0.0):
    if min_delta < 0:
        raise ValueError("min_delta must be >= 0")
    if best is None:
        return True
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError("mode must be 'min' or 'max'")
