import torch
from torch import nn

from vgl.dataloading import DataLoader, FullGraphSampler, ListDataset, SampleRecord
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


def test_many_graph_graph_classification_runs():
    samples = [
        SampleRecord(
            graph=Graph.homo(
                edge_index=torch.tensor([[0], [1]]),
                x=torch.randn(2, 4),
                y=torch.tensor([1]),
            ),
            metadata={},
            sample_id="g1",
        ),
        SampleRecord(
            graph=Graph.homo(
                edge_index=torch.tensor([[0], [1]]),
                x=torch.randn(2, 4),
                y=torch.tensor([0]),
            ),
            metadata={},
            sample_id="g2",
        ),
    ]
    dataset = ListDataset(samples)
    loader = DataLoader(
        dataset=dataset,
        sampler=FullGraphSampler(),
        batch_size=2,
        label_source="graph",
        label_key="y",
    )
    trainer = Trainer(
        model=TinyGraphClassifier(),
        task=GraphClassificationTask(target="y", label_source="graph"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(loader)

    assert result["epochs"] == 1


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


def test_many_graph_hetero_graph_classification_runs():
    samples = [
        SampleRecord(
            graph=Graph.hetero(
                nodes={
                    "paper": {"x": torch.randn(2, 4)},
                    "author": {"x": torch.randn(1, 4)},
                },
                edges={
                    ("author", "writes", "paper"): {
                        "edge_index": torch.tensor([[0, 0], [0, 1]]),
                    },
                    ("paper", "written_by", "author"): {
                        "edge_index": torch.tensor([[0, 1], [0, 0]]),
                    },
                },
            ),
            metadata={"label": 1},
            sample_id="h1",
        ),
        SampleRecord(
            graph=Graph.hetero(
                nodes={
                    "paper": {"x": torch.randn(3, 4)},
                    "author": {"x": torch.randn(2, 4)},
                },
                edges={
                    ("author", "writes", "paper"): {
                        "edge_index": torch.tensor([[0, 1], [1, 2]]),
                    },
                    ("paper", "written_by", "author"): {
                        "edge_index": torch.tensor([[1, 2], [0, 1]]),
                    },
                },
            ),
            metadata={"label": 0},
            sample_id="h2",
        ),
    ]
    dataset = ListDataset(samples)
    loader = DataLoader(
        dataset=dataset,
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

    assert result["epochs"] == 1

