from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, ListDataset, NodeNeighborSampler
from vgl.engine import JSONLinesLogger, Trainer
from vgl.graph import Graph
from vgl.tasks import NodeClassificationTask


class TinyHomoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        if hasattr(graph, "graph"):
            graph = graph.graph
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


def _seed_dataset(graph, mask_key):
    seeds = graph.ndata[mask_key].nonzero(as_tuple=False).view(-1).tolist()
    return ListDataset(
        [
            (graph, {"seed": int(seed), "sample_id": f"{mask_key}:{seed}"})
            for seed in seeds
        ]
    )


def build_demo_loaders():
    graph = build_demo_graph()
    train_loader = DataLoader(
        dataset=_seed_dataset(graph, "train_mask"),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=2,
    )
    val_loader = DataLoader(
        dataset=_seed_dataset(graph, "val_mask"),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=2,
    )
    test_loader = DataLoader(
        dataset=_seed_dataset(graph, "test_mask"),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=2,
    )
    return train_loader, val_loader, test_loader


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_loader, val_loader, test_loader = build_demo_loaders()
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
            loggers=[JSONLinesLogger(Path(tmp_dir) / "train.jsonl", flush=True)],
            log_every_n_steps=1,
        )
        history = trainer.fit(train_loader, val_data=val_loader)
        test_result = trainer.test(test_loader)
        result = {
            "epochs": history["epochs"],
            "best_epoch": history["best_epoch"],
            "test_accuracy": test_result["accuracy"],
            "log_path": str(Path(tmp_dir) / "train.jsonl"),
        }
        print(result)
        return result


if __name__ == "__main__":
    main()
