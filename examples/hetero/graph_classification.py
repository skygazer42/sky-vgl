from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, FullGraphSampler, ListDataset, SampleRecord
from vgl.engine import Trainer
from vgl.graph import Graph
from vgl.nn.readout import global_mean_pool
from vgl.tasks import GraphClassificationTask


class TinyHeteroGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.paper_encoder = nn.Linear(4, 4)
        self.author_encoder = nn.Linear(4, 4)
        self.head = nn.Linear(8, 2)

    def forward(self, batch):
        paper_x = torch.cat([graph.nodes["paper"].x for graph in batch.graphs], dim=0)
        author_x = torch.cat([graph.nodes["author"].x for graph in batch.graphs], dim=0)
        paper_repr = global_mean_pool(self.paper_encoder(paper_x), batch.graph_index_by_type["paper"])
        author_repr = global_mean_pool(self.author_encoder(author_x), batch.graph_index_by_type["author"])
        return self.head(torch.cat([paper_repr, author_repr], dim=-1))


def _hetero_graph(num_papers, num_authors):
    src = torch.arange(num_authors, dtype=torch.long)
    dst = torch.remainder(torch.arange(num_authors, dtype=torch.long) + 1, num_papers)
    return Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(num_papers, 4)},
            "author": {"x": torch.randn(num_authors, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.stack([src, dst], dim=0),
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.stack([dst, src], dim=0),
            },
        },
    )


def main():
    samples = [
        SampleRecord(graph=_hetero_graph(2, 1), metadata={"label": 1}, sample_id="h1"),
        SampleRecord(graph=_hetero_graph(3, 2), metadata={"label": 0}, sample_id="h2"),
    ]
    loader = DataLoader(
        dataset=ListDataset(samples),
        sampler=FullGraphSampler(),
        batch_size=2,
        label_source="metadata",
        label_key="label",
    )
    trainer = Trainer(
        model=TinyHeteroGraphClassifier(),
        task=GraphClassificationTask(target="label", label_source="metadata"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    result = trainer.fit(loader)
    print(result)


if __name__ == "__main__":
    main()
