import torch

from gnn.train.tasks import GraphClassificationTask


class FakeBatch:
    labels = torch.tensor([1, 0])
    metadata = [{"label": 1}, {"label": 0}]


def test_graph_classification_task_uses_batch_labels():
    task = GraphClassificationTask(target="y", label_source="graph")
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    assert loss.ndim == 0


def test_graph_classification_task_uses_metadata_labels():
    task = GraphClassificationTask(target="label", label_source="metadata")
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    assert loss.ndim == 0
