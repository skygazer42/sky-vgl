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
        history = cls(
            epochs=state["epochs"],
            monitor=state["monitor"],
            run_name=state.get("run_name"),
            root_dir=state.get("root_dir"),
            fast_dev_run=state.get("fast_dev_run", False),
            profiler=state.get("profiler"),
        )
        history.update(dict(state))
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
        self["profile"] = None if profile is None else dict(profile)
