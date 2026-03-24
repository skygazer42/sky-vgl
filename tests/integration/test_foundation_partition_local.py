from dataclasses import dataclass

import torch
from torch import nn

from vgl import Graph
from vgl.dataloading import DataLoader, ListDataset, NodeNeighborSampler
from vgl.distributed import LocalGraphShard, LocalSamplingCoordinator
from vgl.distributed import write_partitioned_graph
from vgl.engine import Trainer
from vgl.tasks import NodeClassificationTask


@dataclass(slots=True)
class RoutedNodeBatch:
    base: object
    shard_routes: tuple
    fetched_x: torch.Tensor

    def __getattr__(self, name):
        return getattr(self.base, name)


class RoutedNodeLoader:
    def __init__(self, base_loader, coordinator, node_ids):
        self.base_loader = base_loader
        self.coordinator = coordinator
        self.node_ids = torch.as_tensor(node_ids, dtype=torch.long)

    def __iter__(self):
        routes = self.coordinator.route_node_ids(self.node_ids)
        fetched_x = self.coordinator.fetch_node_features(("node", "node", "x"), self.node_ids).values
        for batch in self.base_loader:
            yield RoutedNodeBatch(batch, routes, fetched_x)


class TinyShardAwareNodeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch):
        assert len(batch.shard_routes) == 2
        assert batch.fetched_x.size(0) == 2
        return self.linear(batch.graph.x)


def test_local_partition_metadata_drives_shard_aware_sampled_training(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.randn(4, 4),
        y=torch.tensor([0, 1, 0, 1]),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    dataset = ListDataset(
        [
            (shards[0].graph, {"seed": 0}),
            (shards[1].graph, {"seed": 1}),
        ]
    )
    base_loader = DataLoader(dataset=dataset, sampler=NodeNeighborSampler(num_neighbors=[-1]), batch_size=2)
    loader = RoutedNodeLoader(base_loader, coordinator, node_ids=torch.tensor([0, 3]))
    trainer = Trainer(
        model=TinyShardAwareNodeClassifier(),
        task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask")),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1
