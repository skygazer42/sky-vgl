import torch
from torch import nn

from vgl import Graph
from vgl.train.tasks import NodeClassificationTask
from vgl.train.trainer import Trainer


class LinearNodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.x)


def test_trainer_runs_single_epoch():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
        val_mask=torch.tensor([True, True]),
        test_mask=torch.tensor([True, True]),
    )
    model = LinearNodeModel()
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
    )
    trainer = Trainer(
        model=model,
        task=task,
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(graph)

    assert history["epochs"] == 1
    assert len(history["train"]) == 1

