from dataclasses import dataclass

import torch
from torch import nn

from vgl.dataloading import DataLoader, ListDataset, NodeNeighborSampler
from vgl.engine import Trainer
from vgl.graph import Graph, GraphSchema
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore
from vgl.tasks import NodeClassificationTask


EDGE_TYPE = ("node", "to", "node")


class TinySampledNodeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch):
        return self.linear(batch.graph.x)


def _storage_backed_graph():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(EDGE_TYPE,),
        node_features={"node": ("x", "y")},
        edge_features={EDGE_TYPE: ("edge_index",)},
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(torch.randn(4, 4)),
            ("node", "node", "y"): InMemoryTensorStore(torch.tensor([0, 1, 0, 1])),
        }
    )
    graph_store = InMemoryGraphStore(
        edges={EDGE_TYPE: torch.tensor([[0, 1, 2], [1, 2, 3]])},
        num_nodes={"node": 4},
    )
    return Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)


def test_storage_backed_graph_sampling_runs_through_public_loader_and_trainer():
    graph = _storage_backed_graph()
    dataset = ListDataset([(graph, {"seed": 0}), (graph, {"seed": 3})])
    loader = DataLoader(dataset=dataset, sampler=NodeNeighborSampler(num_neighbors=[-1]), batch_size=2)
    trainer = Trainer(
        model=TinySampledNodeClassifier(),
        task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask")),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1
