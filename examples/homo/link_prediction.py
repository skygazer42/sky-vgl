from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, FullGraphSampler, LinkPredictionRecord, ListDataset
from vgl.engine import Trainer
from vgl.graph import Graph
from vgl.tasks import LinkPredictionTask


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


def build_demo_loader():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )
    samples = [
        LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
        LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
    ]
    return DataLoader(
        dataset=ListDataset(samples),
        sampler=FullGraphSampler(),
        batch_size=2,
    )


def main():
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    result = trainer.fit(build_demo_loader())
    print(result)
    return result


if __name__ == "__main__":
    main()
