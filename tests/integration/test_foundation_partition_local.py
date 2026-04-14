from dataclasses import dataclass

import torch
from torch import nn

from vgl import Graph
from vgl.dataloading import (
    DataLoader,
    LinkNeighborSampler,
    LinkPredictionRecord,
    ListDataset,
    NodeNeighborSampler,
    TemporalEventRecord,
    TemporalNeighborSampler,
)
from vgl.distributed import LocalGraphShard, LocalSamplingCoordinator, StoreBackedSamplingCoordinator
from vgl.distributed import write_partitioned_graph
from vgl.engine import Trainer
from vgl.tasks import LinkPredictionTask, NodeClassificationTask, TemporalEventPredictionTask


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


def test_store_backed_partition_metadata_drives_shard_aware_sampled_training(tmp_path):
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
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
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


def test_store_backed_temporal_partition_metadata_drives_shard_aware_sampled_training(tmp_path):
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
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
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


def test_store_backed_feature_fetch_preserves_order(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        y=torch.tensor([0, 1]),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=1)
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)

    ids = torch.tensor([1, 0])
    fetched = coordinator.fetch_node_features(("node", "node", "x"), ids).values

    assert fetched.shape == (2, 2)
    assert torch.allclose(fetched[0], graph.x[1])
    assert torch.allclose(fetched[1], graph.x[0])


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


def test_store_backed_multi_relation_partition_metadata_drives_shard_aware_sampled_training(tmp_path):
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
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
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


def test_store_backed_hetero_partition_metadata_drives_shard_aware_sampled_training(tmp_path):
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
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
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


def test_store_backed_hetero_partition_edge_feature_routing_round_trips_across_shards(tmp_path):
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
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)

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



class TinyCoordinatorFeatureAlignedNodeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, batch):
        assert torch.equal(batch.graph.x.view(-1), batch.graph.n_id.to(dtype=batch.graph.x.dtype))
        return self.linear(batch.graph.x)


def test_local_partition_sampled_training_uses_public_loader_path_for_coordinator_feature_fetch(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2], [1, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, True, True, True]),
        test_mask=torch.tensor([True, True, True, True]),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = DataLoader(
        dataset=ListDataset(
            [
                (shards[0].graph, {"seed": 0, "sample_id": "part0"}),
                (shards[1].graph, {"seed": 1, "sample_id": "part1"}),
            ]
        ),
        sampler=NodeNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=2,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyCoordinatorFeatureAlignedNodeClassifier(),
        task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask")),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_training_uses_public_loader_path_for_coordinator_feature_fetch(
    tmp_path,
):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2], [1, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, True, True, True]),
        test_mask=torch.tensor([True, True, True, True]),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset(
            [
                (shards[0].graph, {"seed": 0, "sample_id": "part0_store"}),
                (shards[1].graph, {"seed": 1, "sample_id": "part1_store"}),
            ]
        ),
        sampler=NodeNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=2,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyCoordinatorFeatureAlignedNodeClassifier(),
        task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask")),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_partition_public_loader_path_matches_local_and_store_backed_coordinator_feature_fetch(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2], [1, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, True, True, True]),
        test_mask=torch.tensor([True, True, True, True]),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    dataset = ListDataset(
        [
            (shards[0].graph, {"seed": 0, "sample_id": "part0_parity"}),
            (shards[1].graph, {"seed": 1, "sample_id": "part1_parity"}),
        ]
    )
    local_loader = DataLoader(
        dataset=dataset,
        sampler=NodeNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=2,
        feature_store=LocalSamplingCoordinator(shards),
    )
    store_backed_loader = DataLoader(
        dataset=dataset,
        sampler=NodeNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=2,
        feature_store=StoreBackedSamplingCoordinator.from_partition_dir(tmp_path),
    )

    local_batch = next(iter(local_loader))
    store_backed_batch = next(iter(store_backed_loader))

    assert torch.equal(local_batch.graph.n_id, store_backed_batch.graph.n_id)
    assert torch.equal(local_batch.graph.edge_index, store_backed_batch.graph.edge_index)
    assert torch.equal(local_batch.graph.x, store_backed_batch.graph.x)
    assert torch.equal(local_batch.seed_index, store_backed_batch.seed_index)
    assert local_batch.metadata == store_backed_batch.metadata



class TinyStitchedPartitionHeteroNodeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, batch):
        assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([1]))
        assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([2]))
        assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[2.0]]))
        assert torch.equal(batch.graph.nodes["author"].x, torch.tensor([[30.0]]))
        assert torch.equal(batch.graph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0], [0]]))
        assert torch.equal(batch.graph.edges[("paper", "written_by", "author")].edge_index, torch.tensor([[0], [0]]))
        assert torch.equal(batch.seed_index, torch.tensor([0]))
        return self.linear(batch.graph.nodes["paper"].x)


def test_local_partition_sampled_training_stitched_hetero_sampling_crosses_partition_boundaries(tmp_path):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                "y": torch.tensor([0, 1, 0, 1]),
                "train_mask": torch.tensor([True, True, True, True]),
                "val_mask": torch.tensor([True, True, True, True]),
                "test_mask": torch.tensor([True, True, True, True]),
            },
            "author": {
                "x": torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
            },
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 2], [0, 1]]),
                "edge_weight": torch.tensor([10.0, 20.0]),
            },
            written_by: {
                "edge_index": torch.tensor([[0, 1], [0, 2]]),
                "edge_weight": torch.tensor([100.0, 200.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = DataLoader(
        dataset=ListDataset([(shards[0].graph, {"seed": 1, "node_type": "paper", "sample_id": "stitched_hetero"})]),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"paper": ("x",), "author": ("x",)},
            edge_feature_names={writes: ("edge_weight",), written_by: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroNodeClassifier(),
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

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_training_stitched_hetero_sampling_crosses_partition_boundaries(
    tmp_path,
):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                "y": torch.tensor([0, 1, 0, 1]),
                "train_mask": torch.tensor([True, True, True, True]),
                "val_mask": torch.tensor([True, True, True, True]),
                "test_mask": torch.tensor([True, True, True, True]),
            },
            "author": {
                "x": torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
            },
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 2], [0, 1]]),
                "edge_weight": torch.tensor([10.0, 20.0]),
            },
            written_by: {
                "edge_index": torch.tensor([[0, 1], [0, 2]]),
                "edge_weight": torch.tensor([100.0, 200.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset(
            [
                (
                    shards[0].graph,
                    {"seed": 1, "node_type": "paper", "sample_id": "stitched_hetero_store"},
                )
            ]
        ),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"paper": ("x",), "author": ("x",)},
            edge_feature_names={writes: ("edge_weight",), written_by: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroNodeClassifier(),
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

    assert history["completed_epochs"] == 1


class TinyStitchedPartitionHeteroNodeBlockClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, batch):
        assert batch.blocks is not None
        assert len(batch.blocks) == 3
        outer_block, middle_block, inner_block = batch.blocks
        assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
        assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0, 1]))
        assert torch.equal(batch.graph.nodes["author"].x.view(-1), torch.tensor([10.0, 30.0]))
        assert torch.equal(batch.graph.nodes["paper"].x.view(-1), torch.tensor([1.0, 2.0]))
        assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1]))
        assert torch.equal(outer_block.src_n_id, torch.tensor([0, 2]))
        assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 2]))
        assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 11.0, 12.0]))
        assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([10.0, 30.0]))
        assert torch.equal(outer_block.dstdata["x"].view(-1), torch.tensor([1.0, 2.0]))
        assert torch.equal(middle_block.dst_n_id, torch.tensor([0]))
        assert torch.equal(middle_block.src_n_id, torch.tensor([0, 2]))
        assert torch.equal(middle_block.edata["e_id"], torch.tensor([0, 2]))
        assert torch.equal(middle_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))
        assert torch.equal(inner_block.dst_n_id, torch.tensor([0]))
        assert torch.equal(inner_block.src_n_id, torch.tensor([0, 2]))
        assert torch.equal(inner_block.edata["e_id"], torch.tensor([0, 2]))
        assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))
        assert torch.equal(batch.seed_index, torch.tensor([0]))
        return self.linear(batch.graph.nodes["paper"].x)


def test_local_partition_sampled_training_stitched_hetero_node_output_blocks_materialize_relation_local_blocks(
    tmp_path,
):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {
                "x": torch.tensor([[1.0], [2.0]]),
                "y": torch.tensor([0, 1]),
                "train_mask": torch.tensor([True, True]),
                "val_mask": torch.tensor([True, True]),
                "test_mask": torch.tensor([True, True]),
            },
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]]),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            written_by: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]]),
                "edge_weight": torch.tensor([100.0, 110.0, 120.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = DataLoader(
        dataset=ListDataset(
            [
                (
                    shards[0].graph,
                    {"seed": 0, "node_type": "paper", "sample_id": "stitched_hetero_node_blocks"},
                )
            ]
        ),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={writes: ("edge_weight",), written_by: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroNodeBlockClassifier(),
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

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_training_stitched_hetero_node_output_blocks_materialize_relation_local_blocks(
    tmp_path,
):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {
                "x": torch.tensor([[1.0], [2.0]]),
                "y": torch.tensor([0, 1]),
                "train_mask": torch.tensor([True, True]),
                "val_mask": torch.tensor([True, True]),
                "test_mask": torch.tensor([True, True]),
            },
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]]),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            written_by: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]]),
                "edge_weight": torch.tensor([100.0, 110.0, 120.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset(
            [
                (
                    shards[0].graph,
                    {"seed": 0, "node_type": "paper", "sample_id": "stitched_hetero_node_blocks_store"},
                )
            ]
        ),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={writes: ("edge_weight",), written_by: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroNodeBlockClassifier(),
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

    assert history["completed_epochs"] == 1



class TinyStitchedPartitionNodeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, batch):
        assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
        assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
        assert torch.equal(batch.graph.x.view(-1), torch.tensor([0.0, 1.0, 2.0]))
        assert torch.equal(batch.seed_index, torch.tensor([1]))
        return self.linear(batch.graph.x)


def test_local_partition_sampled_training_stitched_sampling_crosses_partition_boundaries(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, True, True, True]),
        test_mask=torch.tensor([True, True, True, True]),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = DataLoader(
        dataset=ListDataset([(shards[0].graph, {"seed": 1, "sample_id": "stitched"})]),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionNodeClassifier(),
        task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask")),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_training_stitched_sampling_crosses_partition_boundaries(
    tmp_path,
):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, True, True, True]),
        test_mask=torch.tensor([True, True, True, True]),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset([(shards[0].graph, {"seed": 1, "sample_id": "stitched_store"})]),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionNodeClassifier(),
        task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask")),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_partition_sampled_node_loader_stitched_sampling_matches_local_and_store_backed(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, True, True, True]),
        test_mask=torch.tensor([True, True, True, True]),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    dataset = ListDataset([(shards[0].graph, {"seed": 1, "sample_id": "stitched_parity"})])
    local_loader = DataLoader(
        dataset=dataset,
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=1,
        feature_store=LocalSamplingCoordinator(shards),
    )
    store_backed_loader = DataLoader(
        dataset=dataset,
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=1,
        feature_store=StoreBackedSamplingCoordinator.from_partition_dir(tmp_path),
    )

    local_batch = next(iter(local_loader))
    store_backed_batch = next(iter(store_backed_loader))

    assert torch.equal(local_batch.graph.n_id, store_backed_batch.graph.n_id)
    assert torch.equal(local_batch.graph.edge_index, store_backed_batch.graph.edge_index)
    assert torch.equal(local_batch.graph.x, store_backed_batch.graph.x)
    assert torch.equal(local_batch.seed_index, store_backed_batch.seed_index)
    assert local_batch.metadata == store_backed_batch.metadata



class TinyStitchedPartitionNodeBlockClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, batch):
        assert batch.blocks is not None
        assert len(batch.blocks) == 2
        outer_block, inner_block = batch.blocks
        assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1, 2]))
        assert torch.equal(outer_block.src_n_id, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([0.0, 1.0, 2.0, 3.0]))
        assert torch.equal(inner_block.dst_n_id, torch.tensor([1]))
        return self.linear(batch.graph.x)



def test_local_partition_sampled_training_stitched_sampling_materializes_blocks(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, True, True, True]),
        test_mask=torch.tensor([True, True, True, True]),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = DataLoader(
        dataset=ListDataset([(shards[0].graph, {"seed": 1, "sample_id": "stitched_blocks"})]),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionNodeBlockClassifier(),
        task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask")),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_training_stitched_sampling_materializes_blocks(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, True, True, True]),
        test_mask=torch.tensor([True, True, True, True]),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset([(shards[0].graph, {"seed": 1, "sample_id": "stitched_blocks_store"})]),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionNodeBlockClassifier(),
        task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask")),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_partition_sampled_node_loader_stitched_blocks_match_local_and_store_backed(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, True, True, True]),
        test_mask=torch.tensor([True, True, True, True]),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    dataset = ListDataset([(shards[0].graph, {"seed": 1, "sample_id": "stitched_blocks_parity"})])
    local_loader = DataLoader(
        dataset=dataset,
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=LocalSamplingCoordinator(shards),
    )
    store_backed_loader = DataLoader(
        dataset=dataset,
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=StoreBackedSamplingCoordinator.from_partition_dir(tmp_path),
    )

    local_batch = next(iter(local_loader))
    store_backed_batch = next(iter(store_backed_loader))

    assert local_batch.blocks is not None
    assert store_backed_batch.blocks is not None
    assert len(local_batch.blocks) == len(store_backed_batch.blocks) == 2
    assert torch.equal(local_batch.graph.n_id, store_backed_batch.graph.n_id)
    assert torch.equal(local_batch.graph.edge_index, store_backed_batch.graph.edge_index)
    assert torch.equal(local_batch.graph.x, store_backed_batch.graph.x)
    assert torch.equal(local_batch.graph.edata["edge_weight"], store_backed_batch.graph.edata["edge_weight"])
    assert torch.equal(local_batch.seed_index, store_backed_batch.seed_index)
    for local_block, store_block in zip(local_batch.blocks, store_backed_batch.blocks):
        assert torch.equal(local_block.dst_n_id, store_block.dst_n_id)
        assert torch.equal(local_block.src_n_id, store_block.src_n_id)
        assert torch.equal(local_block.edata["edge_weight"], store_block.edata["edge_weight"])



class TinyStitchedPartitionTemporalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(3, 2)

    def forward(self, batch):
        history = batch.history_graph(0)
        edge_type = next(iter(history.edges))
        assert torch.equal(batch.timestamp, torch.tensor([4]))
        assert torch.equal(history.n_id, torch.tensor([0, 1, 2]))
        assert torch.equal(history.edge_index, torch.tensor([[0, 1], [1, 2]]))
        assert torch.equal(history.edges[edge_type].timestamp, torch.tensor([1, 3]))
        assert torch.equal(history.x.view(-1), torch.tensor([0.0, 1.0, 2.0]))
        assert torch.equal(batch.src_index, torch.tensor([0]))
        assert torch.equal(batch.dst_index, torch.tensor([1]))
        src_x = history.x[batch.src_index]
        dst_x = history.x[batch.dst_index]
        time_x = batch.timestamp.to(dtype=history.x.dtype).unsqueeze(-1)
        return self.scorer(torch.cat([src_x, dst_x, time_x], dim=-1))


def test_local_partition_sampled_temporal_training_stitched_temporal_sampling_crosses_partition_boundaries(tmp_path):
    edge_type = ("node", "interacts", "node")
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            edge_type: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                "timestamp": torch.tensor([1, 3, 5]),
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
    loader = DataLoader(
        dataset=ListDataset(
            [
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    timestamp=4,
                    label=1,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionTemporalPredictor(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_temporal_training_stitched_temporal_sampling_crosses_partition_boundaries(
    tmp_path,
):
    edge_type = ("node", "interacts", "node")
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            edge_type: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset(
            [
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    timestamp=4,
                    label=1,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionTemporalPredictor(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_partition_sampled_temporal_loader_stitched_sampling_matches_local_and_store_backed(tmp_path):
    edge_type = ("node", "interacts", "node")
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            edge_type: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    dataset = ListDataset(
        [
            TemporalEventRecord(
                graph=shards[0].graph,
                src_index=0,
                dst_index=1,
                timestamp=4,
                label=1,
            )
        ]
    )
    local_loader = DataLoader(
        dataset=dataset,
        sampler=TemporalNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=1,
        feature_store=LocalSamplingCoordinator(shards),
    )
    store_backed_loader = DataLoader(
        dataset=dataset,
        sampler=TemporalNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=1,
        feature_store=StoreBackedSamplingCoordinator.from_partition_dir(tmp_path),
    )

    local_batch = next(iter(local_loader))
    store_backed_batch = next(iter(store_backed_loader))
    local_history = local_batch.history_graph(0)
    store_backed_history = store_backed_batch.history_graph(0)

    assert torch.equal(local_batch.timestamp, store_backed_batch.timestamp)
    assert torch.equal(local_history.n_id, store_backed_history.n_id)
    assert torch.equal(local_history.edge_index, store_backed_history.edge_index)
    assert torch.equal(local_history.edges[edge_type].timestamp, store_backed_history.edges[edge_type].timestamp)
    assert torch.equal(local_history.x, store_backed_history.x)
    assert torch.equal(local_batch.src_index, store_backed_batch.src_index)
    assert torch.equal(local_batch.dst_index, store_backed_batch.dst_index)



class TinyStitchedPartitionLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(2, 1)

    def forward(self, batch):
        assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
        assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
        assert torch.equal(batch.graph.x.view(-1), torch.tensor([0.0, 1.0, 2.0]))
        assert torch.equal(batch.src_index, torch.tensor([0]))
        assert torch.equal(batch.dst_index, torch.tensor([1]))
        src_x = batch.graph.x[batch.src_index]
        dst_x = batch.graph.x[batch.dst_index]
        return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)


def test_local_partition_sampled_link_training_stitched_link_sampling_crosses_partition_boundaries(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_link_training_stitched_link_sampling_crosses_partition_boundaries(
    tmp_path,
):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_partition_sampled_link_loader_stitched_link_sampling_matches_local_and_store_backed(
    tmp_path,
):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    dataset = ListDataset(
        [
            LinkPredictionRecord(
                graph=shards[0].graph,
                src_index=0,
                dst_index=1,
                label=1,
            )
        ]
    )
    local_loader = DataLoader(
        dataset=dataset,
        sampler=LinkNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=1,
        feature_store=LocalSamplingCoordinator(shards),
    )
    store_backed_loader = DataLoader(
        dataset=dataset,
        sampler=LinkNeighborSampler(num_neighbors=[-1], node_feature_names=("x",)),
        batch_size=1,
        feature_store=StoreBackedSamplingCoordinator.from_partition_dir(tmp_path),
    )

    local_batch = next(iter(local_loader))
    store_backed_batch = next(iter(store_backed_loader))

    assert torch.equal(local_batch.graph.n_id, store_backed_batch.graph.n_id)
    assert torch.equal(local_batch.graph.edge_index, store_backed_batch.graph.edge_index)
    assert torch.equal(local_batch.graph.x, store_backed_batch.graph.x)
    assert torch.equal(local_batch.src_index, store_backed_batch.src_index)
    assert torch.equal(local_batch.dst_index, store_backed_batch.dst_index)
    assert torch.equal(local_batch.labels, store_backed_batch.labels)



class TinyStitchedPartitionLinkBlockPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(2, 1)

    def forward(self, batch):
        assert batch.blocks is not None
        assert len(batch.blocks) == 2
        outer_block, inner_block = batch.blocks
        assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 3]]))
        assert torch.equal(batch.graph.x.view(-1), torch.tensor([0.0, 1.0, 2.0, 3.0]))
        assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([10.0, 20.0, 30.0]))
        assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1, 2]))
        assert torch.equal(outer_block.src_n_id, torch.tensor([0, 1, 2]))
        assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([0.0, 1.0, 2.0]))
        assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 20.0]))
        assert torch.equal(inner_block.dst_n_id, torch.tensor([0, 1]))
        assert torch.equal(inner_block.src_n_id, torch.tensor([0, 1]))
        assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0]))
        assert torch.equal(batch.src_index, torch.tensor([0]))
        assert torch.equal(batch.dst_index, torch.tensor([1]))
        src_x = batch.graph.x[batch.src_index]
        dst_x = batch.graph.x[batch.dst_index]
        return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)



def test_local_partition_sampled_link_training_stitched_sampling_materializes_blocks(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionLinkBlockPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_link_training_stitched_sampling_materializes_blocks(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionLinkBlockPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_partition_sampled_link_loader_stitched_blocks_match_local_and_store_backed(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    dataset = ListDataset(
        [
            LinkPredictionRecord(
                graph=shards[0].graph,
                src_index=0,
                dst_index=1,
                label=1,
            )
        ]
    )
    local_loader = DataLoader(
        dataset=dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=LocalSamplingCoordinator(shards),
    )
    store_backed_loader = DataLoader(
        dataset=dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=StoreBackedSamplingCoordinator.from_partition_dir(tmp_path),
    )

    local_batch = next(iter(local_loader))
    store_backed_batch = next(iter(store_backed_loader))

    assert local_batch.blocks is not None
    assert store_backed_batch.blocks is not None
    assert len(local_batch.blocks) == len(store_backed_batch.blocks) == 2
    assert torch.equal(local_batch.graph.n_id, store_backed_batch.graph.n_id)
    assert torch.equal(local_batch.graph.edge_index, store_backed_batch.graph.edge_index)
    assert torch.equal(local_batch.graph.x, store_backed_batch.graph.x)
    assert torch.equal(local_batch.graph.edata["edge_weight"], store_backed_batch.graph.edata["edge_weight"])
    assert torch.equal(local_batch.src_index, store_backed_batch.src_index)
    assert torch.equal(local_batch.dst_index, store_backed_batch.dst_index)
    assert torch.equal(local_batch.labels, store_backed_batch.labels)
    for local_block, store_block in zip(local_batch.blocks, store_backed_batch.blocks):
        assert torch.equal(local_block.dst_n_id, store_block.dst_n_id)
        assert torch.equal(local_block.src_n_id, store_block.src_n_id)
        assert torch.equal(local_block.edata["edge_weight"], store_block.edata["edge_weight"])



class TinyStitchedPartitionHeteroLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(2, 1)

    def forward(self, batch):
        writes = ("author", "writes", "paper")
        written_by = ("paper", "written_by", "author")
        assert batch.edge_type == writes
        assert batch.src_node_type == "author"
        assert batch.dst_node_type == "paper"
        assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
        assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0]))
        assert torch.equal(batch.graph.nodes["author"].x.view(-1), torch.tensor([10.0, 30.0]))
        assert torch.equal(batch.graph.nodes["paper"].x.view(-1), torch.tensor([1.0]))
        assert torch.equal(batch.graph.edges[writes].edge_index, torch.tensor([[0, 1], [0, 0]]))
        assert torch.equal(batch.graph.edges[written_by].edge_index, torch.tensor([[0, 0], [0, 1]]))
        assert torch.equal(batch.src_index, torch.tensor([0]))
        assert torch.equal(batch.dst_index, torch.tensor([0]))
        src_x = batch.graph.nodes["author"].x[batch.src_index]
        dst_x = batch.graph.nodes["paper"].x[batch.dst_index]
        return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)


def test_local_partition_sampled_link_training_stitched_hetero_link_sampling_crosses_partition_boundaries(tmp_path):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 2], [0, 0]])},
            written_by: {"edge_index": torch.tensor([[0, 0], [0, 2]])},
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    label=1,
                    edge_type=writes,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_link_training_stitched_hetero_link_sampling_crosses_partition_boundaries(
    tmp_path,
):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 2], [0, 0]])},
            written_by: {"edge_index": torch.tensor([[0, 0], [0, 2]])},
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    label=1,
                    edge_type=writes,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_partition_sampled_link_loader_stitched_hetero_link_sampling_matches_local_and_store_backed(
    tmp_path,
):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 2], [0, 0]])},
            written_by: {"edge_index": torch.tensor([[0, 0], [0, 2]])},
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    dataset = ListDataset(
        [
            LinkPredictionRecord(
                graph=shards[0].graph,
                src_index=0,
                dst_index=0,
                label=1,
                edge_type=writes,
            )
        ]
    )
    local_loader = DataLoader(
        dataset=dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
        ),
        batch_size=1,
        feature_store=LocalSamplingCoordinator(shards),
    )
    store_backed_loader = DataLoader(
        dataset=dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
        ),
        batch_size=1,
        feature_store=StoreBackedSamplingCoordinator.from_partition_dir(tmp_path),
    )

    local_batch = next(iter(local_loader))
    store_backed_batch = next(iter(store_backed_loader))

    assert local_batch.edge_type == store_backed_batch.edge_type
    assert local_batch.src_node_type == store_backed_batch.src_node_type
    assert local_batch.dst_node_type == store_backed_batch.dst_node_type
    assert torch.equal(local_batch.graph.nodes["author"].n_id, store_backed_batch.graph.nodes["author"].n_id)
    assert torch.equal(local_batch.graph.nodes["paper"].n_id, store_backed_batch.graph.nodes["paper"].n_id)
    assert torch.equal(local_batch.graph.nodes["author"].x, store_backed_batch.graph.nodes["author"].x)
    assert torch.equal(local_batch.graph.nodes["paper"].x, store_backed_batch.graph.nodes["paper"].x)
    assert torch.equal(local_batch.graph.edges[writes].edge_index, store_backed_batch.graph.edges[writes].edge_index)
    assert torch.equal(
        local_batch.graph.edges[written_by].edge_index,
        store_backed_batch.graph.edges[written_by].edge_index,
    )
    assert torch.equal(local_batch.src_index, store_backed_batch.src_index)
    assert torch.equal(local_batch.dst_index, store_backed_batch.dst_index)
    assert torch.equal(local_batch.labels, store_backed_batch.labels)


class TinyStitchedPartitionHeteroLinkBlockPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(2, 1)

    def forward(self, batch):
        writes = ("author", "writes", "paper")
        assert batch.edge_type == writes
        assert batch.blocks is not None
        assert len(batch.blocks) == 2
        outer_block, inner_block = batch.blocks
        assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
        assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0, 1]))
        assert torch.equal(batch.graph.nodes["author"].x.view(-1), torch.tensor([10.0, 30.0]))
        assert torch.equal(batch.graph.nodes["paper"].x.view(-1), torch.tensor([1.0, 2.0]))
        assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1]))
        assert torch.equal(outer_block.src_n_id, torch.tensor([0, 2]))
        assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 2]))
        assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 11.0, 12.0]))
        assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([10.0, 30.0]))
        assert torch.equal(outer_block.dstdata["x"].view(-1), torch.tensor([1.0, 2.0]))
        assert torch.equal(inner_block.dst_n_id, torch.tensor([0]))
        assert torch.equal(inner_block.src_n_id, torch.tensor([0, 2]))
        assert torch.equal(inner_block.edata["e_id"], torch.tensor([0, 2]))
        assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))
        src_x = batch.graph.nodes["author"].x[batch.src_index]
        dst_x = batch.graph.nodes["paper"].x[batch.dst_index]
        return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)


def test_local_partition_sampled_link_training_stitched_hetero_sampling_materializes_relation_local_blocks(tmp_path):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]]),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            written_by: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]]),
                "edge_weight": torch.tensor([100.0, 110.0, 120.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    label=1,
                    edge_type=writes,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={writes: ("edge_weight",), written_by: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroLinkBlockPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_link_training_stitched_hetero_sampling_materializes_relation_local_blocks(
    tmp_path,
):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]]),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            written_by: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]]),
                "edge_weight": torch.tensor([100.0, 110.0, 120.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    label=1,
                    edge_type=writes,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={writes: ("edge_weight",), written_by: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroLinkBlockPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_partition_sampled_link_loader_stitched_hetero_blocks_match_local_and_store_backed(tmp_path):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]]),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            written_by: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]]),
                "edge_weight": torch.tensor([100.0, 110.0, 120.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    dataset = ListDataset(
        [
            LinkPredictionRecord(
                graph=shards[0].graph,
                src_index=0,
                dst_index=0,
                label=1,
                edge_type=writes,
            )
        ]
    )
    local_loader = DataLoader(
        dataset=dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={writes: ("edge_weight",), written_by: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=LocalSamplingCoordinator(shards),
    )
    store_backed_loader = DataLoader(
        dataset=dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={writes: ("edge_weight",), written_by: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=StoreBackedSamplingCoordinator.from_partition_dir(tmp_path),
    )

    local_batch = next(iter(local_loader))
    store_backed_batch = next(iter(store_backed_loader))

    assert local_batch.blocks is not None
    assert store_backed_batch.blocks is not None
    assert len(local_batch.blocks) == len(store_backed_batch.blocks) == 2
    assert local_batch.edge_type == store_backed_batch.edge_type
    assert torch.equal(local_batch.graph.nodes["author"].n_id, store_backed_batch.graph.nodes["author"].n_id)
    assert torch.equal(local_batch.graph.nodes["paper"].n_id, store_backed_batch.graph.nodes["paper"].n_id)
    assert torch.equal(local_batch.graph.nodes["author"].x, store_backed_batch.graph.nodes["author"].x)
    assert torch.equal(local_batch.graph.nodes["paper"].x, store_backed_batch.graph.nodes["paper"].x)
    assert torch.equal(local_batch.src_index, store_backed_batch.src_index)
    assert torch.equal(local_batch.dst_index, store_backed_batch.dst_index)
    assert torch.equal(local_batch.labels, store_backed_batch.labels)
    for local_block, store_block in zip(local_batch.blocks, store_backed_batch.blocks):
        assert torch.equal(local_block.dst_n_id, store_block.dst_n_id)
        assert torch.equal(local_block.src_n_id, store_block.src_n_id)
        assert torch.equal(local_block.edata["e_id"], store_block.edata["e_id"])
        assert torch.equal(local_block.edata["edge_weight"], store_block.edata["edge_weight"])



class TinyStitchedPartitionHeteroTemporalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(3, 2)

    def forward(self, batch):
        history = batch.history_graph(0)
        assert batch.edge_type == ("author", "writes", "paper")
        assert batch.src_node_type == "author"
        assert batch.dst_node_type == "paper"
        assert torch.equal(batch.timestamp, torch.tensor([4]))
        assert torch.equal(history.nodes["author"].n_id, torch.tensor([0, 2]))
        assert torch.equal(history.nodes["paper"].n_id, torch.tensor([0]))
        assert torch.equal(history.nodes["author"].x.view(-1), torch.tensor([10.0, 30.0]))
        assert torch.equal(history.nodes["paper"].x.view(-1), torch.tensor([1.0]))
        assert torch.equal(
            history.edges[("author", "writes", "paper")].edge_index,
            torch.tensor([[0, 1], [0, 0]]),
        )
        assert torch.equal(
            history.edges[("author", "writes", "paper")].timestamp,
            torch.tensor([1, 3]),
        )
        assert torch.equal(batch.src_index, torch.tensor([0]))
        assert torch.equal(batch.dst_index, torch.tensor([0]))
        src_x = history.nodes["author"].x[batch.src_index]
        dst_x = history.nodes["paper"].x[batch.dst_index]
        time_x = batch.timestamp.to(dtype=src_x.dtype).unsqueeze(-1)
        return self.scorer(torch.cat([src_x, dst_x, time_x], dim=-1))


def test_local_partition_sampled_temporal_training_stitched_hetero_temporal_sampling_crosses_partition_boundaries(tmp_path):
    writes = ("author", "writes", "paper")
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 2, 1], [0, 0, 1]]),
                "timestamp": torch.tensor([1, 3, 6]),
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
    loader = DataLoader(
        dataset=ListDataset(
            [
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    timestamp=4,
                    label=1,
                    edge_type=writes,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroTemporalPredictor(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1


def test_store_backed_partition_sampled_temporal_training_stitched_hetero_temporal_sampling_crosses_partition_boundaries(
    tmp_path,
):
    writes = ("author", "writes", "paper")
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 2, 1], [0, 0, 1]]),
                "timestamp": torch.tensor([1, 3, 6]),
            }
        },
        time_attr="timestamp",
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    loader = DataLoader(
        dataset=ListDataset(
            [
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    timestamp=4,
                    label=1,
                    edge_type=writes,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )
    trainer = Trainer(
        model=TinyStitchedPartitionHeteroTemporalPredictor(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1
