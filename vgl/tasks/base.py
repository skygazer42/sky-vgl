class Task:
    def loss(self, graph, predictions, stage):
        raise NotImplementedError

    def targets(self, batch, stage):
        raise NotImplementedError

    def predictions_for_metrics(self, batch, predictions, stage):
        del batch, stage
        return predictions
