from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl import (
    APPNPConv,
    ChebConv,
    GATv2Conv,
    GINConv,
    Graph,
    NodeClassificationTask,
    SGConv,
    TAGConv,
    Trainer,
)


def build_demo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
        y=torch.tensor([0, 1, 0]),
        train_mask=torch.tensor([True, True, True]),
        val_mask=torch.tensor([True, True, True]),
        test_mask=torch.tensor([True, True, True]),
    )


class TinyConvModel(nn.Module):
    def __init__(self, conv, hidden_channels):
        super().__init__()
        self.conv = conv
        self.head = nn.Linear(hidden_channels, 2)

    def forward(self, graph):
        return self.head(self.conv(graph))


def run_one(name, conv, hidden_channels):
    graph = build_demo_graph()
    trainer = Trainer(
        model=TinyConvModel(conv, hidden_channels),
        task=NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            metrics=["accuracy"],
        ),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    history = trainer.fit(graph, val_data=graph)
    return {"name": name, "loss": history["train"][-1]["loss"]}


def main():
    results = [
        run_one("gin", GINConv(in_channels=4, out_channels=4), 4),
        run_one("gatv2", GATv2Conv(in_channels=4, out_channels=4, heads=2, concat=False), 4),
        run_one("appnp", APPNPConv(in_channels=4, out_channels=4, steps=2, alpha=0.1), 4),
        run_one("tag", TAGConv(in_channels=4, out_channels=4, k=2), 4),
        run_one("sg", SGConv(in_channels=4, out_channels=4, k=2), 4),
        run_one("cheb", ChebConv(in_channels=4, out_channels=4, k=3), 4),
    ]
    print(results)
    return results


if __name__ == "__main__":
    main()
