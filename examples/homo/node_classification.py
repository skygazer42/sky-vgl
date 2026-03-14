from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl import Graph, NodeClassificationTask, Trainer


class TinyHomoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.x)


def build_demo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
        val_mask=torch.tensor([True, True]),
        test_mask=torch.tensor([True, True]),
    )


def main():
    graph = build_demo_graph()
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            model=TinyHomoModel(),
            task=NodeClassificationTask(
                target="y",
                split=("train_mask", "val_mask", "test_mask"),
                metrics=["accuracy"],
            ),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=2,
            monitor="val_accuracy",
            save_best_path=Path(tmp_dir) / "best.pt",
        )
        history = trainer.fit(graph, val_data=graph)
        test_result = trainer.test(graph)
        result = {
            "epochs": history["epochs"],
            "best_epoch": history["best_epoch"],
            "test_accuracy": test_result["accuracy"],
        }
        print(result)
        return result


if __name__ == "__main__":
    main()

