import torch
from torch import nn

from vgl import DatasetManifest, DatasetSplit, OnDiskGraphDataset
from vgl.dataloading import DataLoader, FullGraphSampler, SampleRecord
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


def test_ondisk_graph_dataset_flows_into_loader_and_trainer(tmp_path):
    graphs = [
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([1])),
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([0])),
    ]
    manifest = DatasetManifest(
        name="toy-graph",
        version="1.0",
        splits=(DatasetSplit("train", size=len(graphs)),),
    )
    OnDiskGraphDataset.write(tmp_path, manifest, graphs)
    dataset = OnDiskGraphDataset(tmp_path).split("train")
    loader = DataLoader(
        dataset=[SampleRecord(graph=dataset[index], metadata={}, sample_id=str(index)) for index in range(len(dataset))],
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

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1
