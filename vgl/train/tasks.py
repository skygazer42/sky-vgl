import torch
import torch.nn.functional as F

from vgl.train.task import Task


class NodeClassificationTask(Task):
    def __init__(self, target, split, loss="cross_entropy", metrics=None, node_type=None):
        self.target = target
        self.train_key, self.val_key, self.test_key = split
        self.loss_name = loss
        self.metrics = metrics or []
        self.node_type = node_type

    def _node_data(self, graph):
        if self.node_type is not None:
            return graph.nodes[self.node_type].data
        if "node" in graph.nodes:
            return graph.nodes["node"].data
        if len(graph.nodes) == 1:
            return next(iter(graph.nodes.values())).data
        raise ValueError("node_type is required for multi-type node classification")

    def loss(self, graph, logits, stage):
        node_data = self._node_data(graph)
        mask = node_data[f"{stage}_mask"]
        target = node_data[self.target]
        return F.cross_entropy(logits[mask], target[mask])


class GraphClassificationTask(Task):
    def __init__(self, target, label_source="graph", loss="cross_entropy", metrics=None):
        self.target = target
        self.label_source = label_source
        self.loss_name = loss
        self.metrics = metrics or []

    def _targets(self, batch):
        if self.label_source == "graph":
            return batch.labels
        if self.label_source == "metadata":
            return batch.labels if getattr(batch, "labels", None) is not None else torch.tensor(
                [item[self.target] for item in batch.metadata]
            )
        if self.label_source == "auto":
            if getattr(batch, "labels", None) is not None:
                return batch.labels
            return torch.tensor([item[self.target] for item in batch.metadata])
        raise ValueError(f"Unsupported label_source: {self.label_source}")

    def loss(self, batch, logits, stage):
        del stage
        target = self._targets(batch)
        return F.cross_entropy(logits, target)


class LinkPredictionTask(Task):
    def __init__(self, target="label", loss="binary_cross_entropy", metrics=None):
        if loss != "binary_cross_entropy":
            raise ValueError(f"Unsupported loss: {loss}")
        self.target = target
        self.loss_name = loss
        self.metrics = metrics or []

    def loss(self, batch, logits, stage):
        del stage
        if logits.ndim == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        if logits.ndim != 1 or logits.size(0) != batch.labels.size(0):
            raise ValueError("LinkPredictionTask expects one logit per candidate edge")
        targets = batch.labels.to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets)


class TemporalEventPredictionTask(Task):
    def __init__(self, target="label", loss="cross_entropy", metrics=None):
        if loss != "cross_entropy":
            raise ValueError(f"Unsupported loss: {loss}")
        self.target = target
        self.loss_name = loss
        self.metrics = metrics or []

    def loss(self, batch, logits, stage):
        del stage
        return F.cross_entropy(logits, batch.labels)

