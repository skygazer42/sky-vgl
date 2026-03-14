from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import torch

from vgl.train.metrics import build_metric


class Trainer:
    def __init__(
        self,
        model,
        task,
        optimizer,
        lr,
        max_epochs,
        metrics=None,
        monitor=None,
        monitor_mode=None,
        save_best_path=None,
    ):
        self.model = model
        self.task = task
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.max_epochs = max_epochs
        metric_specs = getattr(task, "metrics", None) if metrics is None else metrics
        self.metric_specs = list(metric_specs or [])
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.save_best_path = Path(save_best_path) if save_best_path is not None else None
        self.best_state_dict = None
        self.best_epoch = None
        self.best_metric = None

    def _batches(self, data):
        if (
            hasattr(data, "nodes")
            or hasattr(data, "graphs")
            or hasattr(data, "labels")
            or hasattr(data, "x")
        ):
            return [data]
        if isinstance(data, Iterable):
            return data
        return [data]

    def _metrics(self):
        return [build_metric(metric) for metric in self.metric_specs]

    def _run_epoch(self, data, stage, training):
        batches = list(self._batches(data))
        if not batches:
            raise ValueError(f"Trainer.{stage} requires at least one batch")

        metrics = self._metrics()
        for metric in metrics:
            metric.reset()

        total_loss = 0.0
        total_items = 0
        self.model.train(training)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in batches:
                if training:
                    self.optimizer.zero_grad()
                predictions = self.model(batch)
                loss = self.task.loss(batch, predictions, stage=stage)
                metric_predictions = self.task.predictions_for_metrics(batch, predictions, stage=stage)
                targets = self.task.targets(batch, stage=stage)
                if metric_predictions.size(0) != targets.size(0):
                    raise ValueError("Task metric predictions and targets must align in batch size")
                if training:
                    loss.backward()
                    self.optimizer.step()

                count = int(targets.size(0))
                total_loss += loss.detach().item() * count
                total_items += count
                for metric in metrics:
                    metric.update(metric_predictions.detach(), targets.detach())

        if total_items == 0:
            raise ValueError(f"Trainer.{stage} requires at least one supervised example")

        summary = {"loss": total_loss / total_items}
        for metric in metrics:
            value = metric.compute()
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric {metric.name} must return a scalar value")
            summary[metric.name] = float(value)
        return summary

    def _resolve_monitor(self, val_data):
        monitor = self.monitor or ("val_loss" if val_data is not None else "train_loss")
        if monitor.startswith("val_") and val_data is None:
            raise ValueError("Trainer monitor requires val_data for val_* keys")
        mode = self.monitor_mode
        if mode is None:
            mode = "min" if monitor.endswith("_loss") else "max"
        if mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'")
        return monitor, mode

    def _monitor_value(self, monitor, train_summary, val_summary):
        stage, _, key = monitor.partition("_")
        source = {"train": train_summary, "val": val_summary}.get(stage)
        if source is None or key not in source:
            raise ValueError(f"Monitor key {monitor} was not produced by the trainer")
        return source[key]

    def _save_best(self):
        if self.save_best_path is None:
            return
        if self.save_best_path.exists() and self.save_best_path.is_dir():
            raise ValueError("save_best_path must be a file path, not a directory")
        self.save_best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.best_state_dict, self.save_best_path)

    def evaluate(self, data, stage="val"):
        return self._run_epoch(data, stage=stage, training=False)

    def test(self, data):
        return self.evaluate(data, stage="test")

    def fit(self, train_data, val_data=None):
        monitor, mode = self._resolve_monitor(val_data)
        history = {
            "epochs": self.max_epochs,
            "train": [],
            "val": [],
            "best_epoch": None,
            "best_metric": None,
            "monitor": monitor,
        }

        for epoch in range(1, self.max_epochs + 1):
            train_summary = self._run_epoch(train_data, stage="train", training=True)
            history["train"].append(train_summary)
            val_summary = None
            if val_data is not None:
                val_summary = self._run_epoch(val_data, stage="val", training=False)
                history["val"].append(val_summary)

            current = self._monitor_value(monitor, train_summary, val_summary)
            improved = self.best_metric is None or (
                current < self.best_metric if mode == "min" else current > self.best_metric
            )
            if improved:
                self.best_metric = float(current)
                self.best_epoch = epoch
                self.best_state_dict = deepcopy(self.model.state_dict())
                self._save_best()

        history["best_epoch"] = self.best_epoch
        history["best_metric"] = self.best_metric
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        return history
