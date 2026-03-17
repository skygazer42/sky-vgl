from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, ListDataset, NodeSeedSubgraphSampler
from vgl.engine import Trainer
from vgl.graph import Graph
from vgl.nn.readout import global_mean_pool
from vgl.tasks import GraphClassificationTask


class TinyGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, batch):
        x = torch.cat([graph.x for graph in batch.graphs], dim=0)
        node_repr = self.encoder(x)
        graph_repr = global_mean_pool(node_repr, batch.graph_index)
        return self.head(graph_repr)


def main():
    source_graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )
    dataset = ListDataset(
        [
            (source_graph, {"seed": 1, "label": 1, "sample_id": "s1", "source_graph_id": "root"}),
            (source_graph, {"seed": 2, "label": 0, "sample_id": "s2", "source_graph_id": "root"}),
        ]
    )
    loader = DataLoader(
        dataset=dataset,
        sampler=NodeSeedSubgraphSampler(),
        batch_size=2,
        label_source="metadata",
        label_key="label",
    )
    trainer = Trainer(
        model=TinyGraphClassifier(),
        task=GraphClassificationTask(target="label", label_source="metadata"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    result = trainer.fit(loader)
    print(result)


if __name__ == "__main__":
    main()
