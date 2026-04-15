PROFILE_TOTAL_KEYS = (
    "batch_materialization_seconds_total",
    "forward_seconds_total",
    "backward_seconds_total",
    "optimizer_step_seconds_total",
    "train_step_seconds_total",
    "train_stage_seconds_total",
    "val_stage_seconds_total",
    "test_stage_seconds_total",
    "sanity_val_stage_seconds_total",
)
PROFILE_COUNT_KEYS = ("train_step_count",)
PROFILE_DERIVED_KEYS = ("train_step_seconds_avg",)


def normalize_profile(profile, *, profiler):
    if profile is None:
        return None
    if profiler != "simple":
        raise ValueError("profile requires profiler='simple'")
    if not isinstance(profile, dict):
        raise ValueError("profile must be a mapping")
    normalized = {}
    for key in PROFILE_TOTAL_KEYS:
        try:
            value = float(profile.get(key, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"profile {key} must be numeric") from exc
        if value < 0.0:
            raise ValueError(f"profile {key} must be >= 0")
        normalized[key] = value
    for key in PROFILE_COUNT_KEYS:
        try:
            value = int(profile.get(key, 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"profile {key} must be an integer") from exc
        if value < 0:
            raise ValueError(f"profile {key} must be >= 0")
        normalized[key] = value
    count = normalized["train_step_count"]
    normalized["train_step_seconds_avg"] = (
        0.0 if count == 0 else float(normalized["train_step_seconds_total"]) / count
    )
    return normalized


class TrainingHistory(dict):
    def __init__(
        self,
        *,
        epochs,
        monitor,
        run_name=None,
        root_dir=None,
        fast_dev_run=False,
        profiler=None,
    ):
        super().__init__(
            {
                "epochs": epochs,
                "train": [],
                "val": [],
                "completed_epochs": 0,
                "best_epoch": None,
                "best_metric": None,
                "monitor": monitor,
                "stopped_early": False,
                "stop_reason": None,
                "fit_elapsed_seconds": None,
                "epoch_elapsed_seconds": [],
                "final_train": None,
                "final_val": None,
                "run_name": None if run_name is None else str(run_name),
                "root_dir": None if root_dir is None else str(root_dir),
                "fast_dev_run": bool(fast_dev_run),
                "sanity_check_passed": False,
                "profiler": profiler,
                "profile": None,
            }
        )

    def state_dict(self):
        return dict(self)

    @classmethod
    def from_state_dict(cls, state):
        if not isinstance(state, dict):
            raise ValueError("history_state must be a mapping")
        required_keys = {"epochs", "monitor"}
        missing_keys = sorted(required_keys - set(state))
        if missing_keys:
            raise ValueError(
                "history_state missing required keys: " + ", ".join(missing_keys)
            )
        try:
            epochs = int(state["epochs"])
        except (TypeError, ValueError) as exc:
            raise ValueError("history_state epochs must be an integer") from exc
        monitor = state["monitor"]
        if not isinstance(monitor, str):
            raise ValueError("history_state monitor must be a string")
        profiler = state.get("profiler")
        if profiler not in {None, "simple"}:
            raise ValueError("history_state profiler must be None or 'simple'")
        history = cls(
            epochs=epochs,
            monitor=monitor,
            run_name=state.get("run_name"),
            root_dir=state.get("root_dir"),
            fast_dev_run=state.get("fast_dev_run", False),
            profiler=profiler,
        )
        history.update(dict(state))
        history["epochs"] = epochs
        history["monitor"] = monitor
        history["run_name"] = None if state.get("run_name") is None else str(state.get("run_name"))
        history["root_dir"] = None if state.get("root_dir") is None else str(state.get("root_dir"))
        history["stopped_early"] = bool(state.get("stopped_early", False))
        history["stop_reason"] = None if state.get("stop_reason") is None else str(state.get("stop_reason"))
        history["fast_dev_run"] = bool(state.get("fast_dev_run", False))
        history["sanity_check_passed"] = bool(state.get("sanity_check_passed", False))
        if history["stop_reason"] is not None and not history["stopped_early"]:
            raise ValueError("history_state stop_reason requires stopped_early")
        history["profiler"] = profiler
        if state.get("profile") is not None and history["profiler"] != "simple":
            raise ValueError("history_state profile requires profiler='simple'")
        history["profile"] = normalize_profile(
            state.get("profile"),
            profiler=history["profiler"],
        )
        try:
            completed_epochs = int(history.get("completed_epochs", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("history_state completed_epochs must be an integer") from exc
        if completed_epochs < 0:
            raise ValueError("history_state completed_epochs must be >= 0")
        if completed_epochs > history["epochs"]:
            raise ValueError("history_state completed_epochs must be <= epochs")
        history["completed_epochs"] = completed_epochs
        train_history = history.get("train", [])
        if len(train_history) != completed_epochs:
            raise ValueError("history_state train history length must match completed_epochs")
        if any(not isinstance(summary, dict) for summary in train_history):
            raise ValueError("history_state train history entries must be mappings")
        history["train"] = [dict(summary) for summary in train_history]
        val_history = history.get("val", [])
        if len(val_history) > completed_epochs:
            raise ValueError("history_state val history length must be <= completed_epochs")
        if any(not isinstance(summary, dict) for summary in val_history):
            raise ValueError("history_state val history entries must be mappings")
        if monitor.startswith("val_") and val_history and len(val_history) != completed_epochs:
            raise ValueError("history_state val history length must match completed_epochs for val monitor")
        history["val"] = [dict(summary) for summary in val_history]
        final_train = history.get("final_train")
        if final_train is not None and not isinstance(final_train, dict):
            raise ValueError("history_state final_train must be a mapping")
        history["final_train"] = None if final_train is None else dict(final_train)
        final_val = history.get("final_val")
        if final_val is not None and not isinstance(final_val, dict):
            raise ValueError("history_state final_val must be a mapping")
        history["final_val"] = None if final_val is None else dict(final_val)
        epoch_elapsed = history.get("epoch_elapsed_seconds", [])
        if len(epoch_elapsed) != completed_epochs:
            raise ValueError("history_state epoch_elapsed_seconds length must match completed_epochs")
        normalized_epoch_elapsed = []
        for elapsed_seconds in epoch_elapsed:
            if elapsed_seconds is None:
                normalized_epoch_elapsed.append(None)
                continue
            try:
                elapsed_seconds = float(elapsed_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "history_state epoch_elapsed_seconds entries must be numeric"
                ) from exc
            if elapsed_seconds < 0.0:
                raise ValueError(
                    "history_state epoch_elapsed_seconds entries must be >= 0"
                )
            normalized_epoch_elapsed.append(elapsed_seconds)
        history["epoch_elapsed_seconds"] = normalized_epoch_elapsed
        fit_elapsed_seconds = history.get("fit_elapsed_seconds")
        if fit_elapsed_seconds is not None:
            try:
                fit_elapsed_seconds = float(fit_elapsed_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError("history_state fit_elapsed_seconds must be numeric") from exc
            if fit_elapsed_seconds < 0.0:
                raise ValueError("history_state fit_elapsed_seconds must be >= 0")
            history["fit_elapsed_seconds"] = fit_elapsed_seconds
        best_epoch = history.get("best_epoch")
        if best_epoch is not None:
            try:
                best_epoch = int(best_epoch)
            except (TypeError, ValueError) as exc:
                raise ValueError("history_state best_epoch must be an integer") from exc
            if best_epoch <= 0:
                raise ValueError("history_state best_epoch must be >= 1")
            if best_epoch > completed_epochs:
                raise ValueError("history_state best_epoch must be <= completed_epochs")
            history["best_epoch"] = best_epoch
        best_metric = history.get("best_metric")
        if best_metric is not None:
            try:
                best_metric = float(best_metric)
            except (TypeError, ValueError) as exc:
                raise ValueError("history_state best_metric must be numeric") from exc
            history["best_metric"] = best_metric
        if best_epoch is not None and history.get("best_metric") is None:
            raise ValueError("history_state best_epoch requires best_metric")
        if history.get("best_metric") is not None and best_epoch is None:
            raise ValueError("history_state best_metric requires best_epoch")
        return history

    def record_epoch(
        self,
        *,
        epoch,
        train_summary,
        val_summary,
        best_epoch,
        best_metric,
        elapsed_seconds=None,
    ):
        self["train"].append(dict(train_summary))
        if val_summary is not None:
            self["val"].append(dict(val_summary))
        self["completed_epochs"] = epoch
        self["best_epoch"] = best_epoch
        self["best_metric"] = best_metric
        self["epoch_elapsed_seconds"].append(
            None if elapsed_seconds is None else float(elapsed_seconds)
        )

    def mark_stopped(self, reason):
        self["stopped_early"] = True
        self["stop_reason"] = reason

    def finalize(
        self,
        *,
        best_epoch,
        best_metric,
        final_train=None,
        final_val=None,
        fit_elapsed_seconds=None,
        profile=None,
    ):
        self["best_epoch"] = best_epoch
        self["best_metric"] = best_metric
        self["final_train"] = None if final_train is None else dict(final_train)
        self["final_val"] = None if final_val is None else dict(final_val)
        self["fit_elapsed_seconds"] = (
            None if fit_elapsed_seconds is None else float(fit_elapsed_seconds)
        )
        self["profile"] = normalize_profile(profile, profiler=self["profiler"])
