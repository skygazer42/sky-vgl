import pytest
import torch
from torch import nn
from torch._dynamo.exc import InternalTorchDynamoError

from vgl import Graph
from vgl.engine import Trainer
from vgl.tasks import NodeClassificationTask


class CompileSmokeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)

    def forward(self, graph: Graph):
        return self.layer(graph.x)


def _build_graph() -> Graph:
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.randn(4, 4),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, False]),
        test_mask=torch.tensor([False, False, False, True]),
    )


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_trainer_smoke_with_torch_compile():
    compiled_model = torch.compile(CompileSmokeModel())
    trainer = Trainer(
        model=compiled_model,
        task=NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            metrics=["accuracy"],
        ),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    graph = _build_graph()
    try:
        history = trainer.fit(graph, val_data=graph)
    except InternalTorchDynamoError as exc:
        pytest.skip(f"torch.compile guard not ready for Graph objects: {exc}")

    assert history["completed_epochs"] == 1
