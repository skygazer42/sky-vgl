def resolve_monitor_mode(monitor, mode=None, *, invalid_mode_message="monitor_mode must be 'min' or 'max'"):
    resolved_mode = mode or ("min" if monitor.endswith("_loss") else "max")
    if resolved_mode not in {"min", "max"}:
        raise ValueError(invalid_mode_message)
    return resolved_mode


def resolve_monitor(monitor, *, has_val_data, mode=None):
    resolved_monitor = monitor or ("val_loss" if has_val_data else "train_loss")
    if resolved_monitor.startswith("val_") and not has_val_data:
        raise ValueError("Trainer monitor requires val_data for val_* keys")
    return resolved_monitor, resolve_monitor_mode(resolved_monitor, mode=mode)


def extract_monitor_value(
    monitor,
    *,
    train_summary,
    val_summary,
    error_subject="Monitor key",
):
    stage, _, key = monitor.partition("_")
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
