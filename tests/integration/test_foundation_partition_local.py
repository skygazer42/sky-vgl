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
    def __init__(self, base_loader, coordinator, node_ids, *, node_type="node", feature_key=None):
        self.base_loader = base_loader
        self.coordinator = coordinator
        self.node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        self.node_type = node_type
        self.feature_key = feature_key or ("node", node_type, "x")

    def __iter__(self):
        routes = self.coordinator.route_node_ids(self.node_ids, node_type=self.node_type)
        fetched_x = self.coordinator.fetch_node_features(self.feature_key, self.node_ids).values
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


class TinyShardAwarePaperClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch):
        assert len(batch.shard_routes) == 2
        assert batch.fetched_x.size(0) == 2
        return self.linear(batch.graph.nodes["paper"].x)


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


def test_local_temporal_partition_metadata_drives_shard_aware_sampled_training(tmp_path):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(4, 4), "y": torch.tensor([0, 1, 0, 1])}},
        edges={
            ("node", "to", "node"): {
                "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
                "timestamp": torch.tensor([3, 5, 7, 11]),
            }
        },
        time_attr="timestamp",
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

    assert all(shard.graph.schema.time_attr == "timestamp" for shard in shards.values())
    assert history["completed_epochs"] == 1


def test_local_multi_relation_partition_metadata_drives_shard_aware_sampled_training(tmp_path):
    follows = ("node", "follows", "node")
    likes = ("node", "likes", "node")
    graph = Graph.hetero(
        nodes={"node": {"x": torch.randn(4, 4), "y": torch.tensor([0, 1, 0, 1])}},
        edges={
            follows: {
                "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            },
            likes: {
                "edge_index": torch.tensor([[1, 0, 3], [0, 1, 2]]),
                "score": torch.tensor([0.5, 0.6, 0.7]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    dataset = ListDataset(
        [
            (shards[0].graph, {"seed": 0, "node_type": "node"}),
            (shards[1].graph, {"seed": 1, "node_type": "node"}),
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

    assert all(set(shard.graph.edges) == {follows, likes} for shard in shards.values())
    assert history["completed_epochs"] == 1


def test_local_hetero_partition_metadata_drives_shard_aware_sampled_training(tmp_path):
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(4, 4)},
            "paper": {
                "x": torch.randn(4, 4),
                "y": torch.tensor([0, 1, 0, 1]),
                "train_mask": torch.tensor([True, True, True, True]),
                "val_mask": torch.tensor([True, True, True, True]),
                "test_mask": torch.tensor([True, True, True, True]),
            },
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 1], [1, 0, 3, 2, 2]]),
                "score": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    dataset = ListDataset(
        [
            (shards[0].graph, {"seed": 0, "node_type": "paper"}),
            (shards[1].graph, {"seed": 1, "node_type": "paper"}),
        ]
    )
    base_loader = DataLoader(dataset=dataset, sampler=NodeNeighborSampler(num_neighbors=[-1]), batch_size=2)
    loader = RoutedNodeLoader(
        base_loader,
        coordinator,
        node_ids=torch.tensor([0, 3]),
        node_type="paper",
    )
    trainer = Trainer(
        model=TinyShardAwarePaperClassifier(),
        task=NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            node_type="paper",
        ),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert all(set(shard.graph.nodes) == {"author", "paper"} for shard in shards.values())
    assert all(set(shard.graph.edges) == {writes, cites} for shard in shards.values())
    assert history["completed_epochs"] == 1


def test_local_hetero_partition_edge_feature_routing_round_trips_across_shards(tmp_path):
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    writes_key = ("edge", writes, "weight")
    cites_key = ("edge", cites, "score")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.arange(8, dtype=torch.float32).view(4, 2)},
            "paper": {"x": torch.arange(10, 18, dtype=torch.float32).view(4, 2)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 1], [1, 0, 3, 2, 2]]),
                "score": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)

    writes_edge_ids = torch.tensor([3, 0, 2])
    cites_edge_ids = torch.tensor([3, 0])

    writes_routes = coordinator.route_edge_ids(writes_edge_ids, edge_type=writes)
    writes_features = coordinator.fetch_edge_features(writes_key, writes_edge_ids)
    cites_features = coordinator.fetch_edge_features(cites_key, cites_edge_ids)
    partition_writes = coordinator.partition_edge_ids(1, edge_type=writes)
    partition_writes_index = coordinator.fetch_partition_edge_index(1, edge_type=writes, global_ids=True)

    assert len(writes_routes) == 2
    assert writes_routes[0].partition_id == 0
    assert torch.equal(writes_routes[0].global_ids, torch.tensor([0]))
    assert torch.equal(writes_routes[0].local_ids, torch.tensor([0]))
    assert writes_routes[1].partition_id == 1
    assert torch.equal(writes_routes[1].global_ids, torch.tensor([3, 2]))
    assert torch.equal(writes_routes[1].local_ids, torch.tensor([1, 0]))
    assert torch.equal(writes_features.index, writes_edge_ids)
    assert torch.equal(writes_features.values, torch.tensor([4.0, 1.0, 3.0]))
    assert torch.equal(cites_features.index, cites_edge_ids)
    assert torch.equal(cites_features.values, torch.tensor([0.4, 0.1]))
    assert torch.equal(partition_writes, torch.tensor([2, 3]))
    assert torch.equal(partition_writes_index, torch.tensor([[2, 3], [3, 2]]))
