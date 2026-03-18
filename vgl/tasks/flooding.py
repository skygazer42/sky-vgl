from vgl.tasks.base import Task


class FloodingTask(Task):
    def __init__(self, base_task, level=0.0):
        if level < 0.0:
            raise ValueError("level must be >= 0")
        self.base_task = base_task
        self.level = float(level)
        paired_loss = getattr(base_task, "paired_loss", None)
        if callable(paired_loss):
            self.paired_loss = self._paired_loss

    def __getattr__(self, name):
        return getattr(self.base_task, name)

    def _flood(self, loss, stage):
        if stage != "train":
            return loss
        return (loss - self.level).abs() + self.level

    def loss(self, batch, predictions, stage):
        return self._flood(self.base_task.loss(batch, predictions, stage), stage)

    def _paired_loss(self, batch, predictions, second_predictions, stage):
        return self._flood(
            self.base_task.paired_loss(batch, predictions, second_predictions, stage),
            stage,
        )

    def targets(self, batch, stage):
        return self.base_task.targets(batch, stage)

    def predictions_for_metrics(self, batch, predictions, stage):
        return self.base_task.predictions_for_metrics(batch, predictions, stage)


__all__ = ["FloodingTask"]
