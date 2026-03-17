class TrainingHistory(dict):
    def __init__(self, *, epochs, monitor):
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
            }
        )

    def state_dict(self):
        return dict(self)

    @classmethod
    def from_state_dict(cls, state):
        history = cls(
            epochs=state["epochs"],
            monitor=state["monitor"],
        )
        history.update(dict(state))
        return history

    def record_epoch(self, *, epoch, train_summary, val_summary, best_epoch, best_metric):
        self["train"].append(dict(train_summary))
        if val_summary is not None:
            self["val"].append(dict(val_summary))
        self["completed_epochs"] = epoch
        self["best_epoch"] = best_epoch
        self["best_metric"] = best_metric

    def mark_stopped(self, reason):
        self["stopped_early"] = True
        self["stop_reason"] = reason

    def finalize(self, *, best_epoch, best_metric):
        self["best_epoch"] = best_epoch
        self["best_metric"] = best_metric
