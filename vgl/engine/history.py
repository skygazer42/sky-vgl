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
    if profiler != "simple" or profile is None:
        return None
    if not isinstance(profile, dict):
        raise ValueError("profile must be a mapping")
    normalized = {key: float(profile.get(key, 0.0)) for key in PROFILE_TOTAL_KEYS}
    for key in PROFILE_COUNT_KEYS:
        normalized[key] = int(profile.get(key, 0))
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
        history = cls(
            epochs=state["epochs"],
            monitor=state["monitor"],
            run_name=state.get("run_name"),
            root_dir=state.get("root_dir"),
            fast_dev_run=state.get("fast_dev_run", False),
            profiler=state.get("profiler"),
        )
        history.update(dict(state))
        history["profile"] = normalize_profile(
            state.get("profile"),
            profiler=history["profiler"],
        )
        completed_epochs = int(history.get("completed_epochs", 0))
        if completed_epochs < 0:
            raise ValueError("history_state completed_epochs must be >= 0")
        if completed_epochs > int(history["epochs"]):
            raise ValueError("history_state completed_epochs must be <= epochs")
        train_history = history.get("train", [])
        if len(train_history) != completed_epochs:
            raise ValueError("history_state train history length must match completed_epochs")
        epoch_elapsed = history.get("epoch_elapsed_seconds", [])
        if len(epoch_elapsed) != completed_epochs:
            raise ValueError("history_state epoch_elapsed_seconds length must match completed_epochs")
        best_epoch = history.get("best_epoch")
        if best_epoch is not None:
            best_epoch = int(best_epoch)
            if best_epoch < 0:
                raise ValueError("history_state best_epoch must be >= 0")
            if best_epoch > completed_epochs:
                raise ValueError("history_state best_epoch must be <= completed_epochs")
            history["best_epoch"] = best_epoch
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
