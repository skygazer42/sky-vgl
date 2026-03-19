from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import (
    CandidateLinkSampler,
    DataLoader,
    FullGraphSampler,
    LinkNeighborSampler,
    LinkPredictionRecord,
    ListDataset,
    UniformNegativeLinkSampler,
)
from vgl.engine import JSONLinesLogger, Trainer
from vgl.graph import Graph
from vgl.tasks import LinkPredictionTask
from vgl.transforms import RandomLinkSplit


class TinyLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.scorer = nn.Linear(8, 1)

    def forward(self, batch):
        node_repr = self.encoder(batch.graph.x)
        src_x = node_repr[batch.src_index]
        dst_x = node_repr[batch.dst_index]
        return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)


def build_demo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 4),
    )


def build_demo_loader():
    graph = build_demo_graph()
    samples = [
        LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
        LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
    ]
    return DataLoader(
        dataset=ListDataset(samples),
        sampler=FullGraphSampler(),
        batch_size=2,
    )


def build_split_demo_loaders():
    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        seed=0,
    )(build_demo_graph())
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=UniformNegativeLinkSampler(num_negatives=2),
        ),
        batch_size=2,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=CandidateLinkSampler(),
        ),
        batch_size=2,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=CandidateLinkSampler(),
        ),
        batch_size=2,
    )
    return train_loader, val_loader, test_loader


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_loader, val_loader, test_loader = build_split_demo_loaders()
        trainer = Trainer(
            model=TinyLinkPredictor(),
            task=LinkPredictionTask(target="label", metrics=["mrr", "filtered_mrr", "filtered_hits@1"]),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=1,
            loggers=[JSONLinesLogger(Path(tmp_dir) / "train.jsonl", flush=True)],
            log_every_n_steps=1,
        )
        history = trainer.fit(train_loader, val_data=val_loader)
        test_result = trainer.test(test_loader)
        result = {
            "history": history,
            "test": test_result,
            "log_path": str(Path(tmp_dir) / "train.jsonl"),
        }
        print(result)
        return result


if __name__ == "__main__":
    main()
