import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import TemporalEventRecord
from vgl.data.sampler import TemporalNeighborSampler
from vgl.distributed import LocalGraphShard, LocalSamplingCoordinator, write_partitioned_graph
from vgl.distributed.coordinator import StoreBackedSamplingCoordinator
from vgl.storage import FeatureStore, InMemoryTensorStore


EDGE_TYPE = ("node", "interacts", "node")


def _store_backed_coordinator(shards):
    root = next(iter(shards.values())).root
    return StoreBackedSamplingCoordinator.from_partition_dir(root)


def _graph():
    return Graph.temporal(
        nodes={"node": {"x": torch.randn(4, 4)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )


def test_temporal_neighbor_sampler_extracts_strict_history_subgraph():
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])

    record = sampler.sample(
        TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1)
    )
    edge_type = EDGE_TYPE

    assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(record.graph.edges[edge_type].edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(record.graph.edges[edge_type].timestamp, torch.tensor([1]))
    assert record.src_index == 1
    assert record.dst_index == 2


def test_temporal_neighbor_sampler_can_limit_history_by_max_events():
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1], max_events=1)

    record = sampler.sample(
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=6, label=1)
    )
    edge_type = EDGE_TYPE

    assert torch.equal(record.graph.n_id, torch.tensor([0, 2]))
    assert torch.equal(record.graph.edges[edge_type].edge_index, torch.tensor([[1], [0]]))
    assert torch.equal(record.graph.edges[edge_type].timestamp, torch.tensor([5]))


def test_loader_routes_temporal_neighbor_sampler_through_plan_execution():
    graph = _graph()
    dataset = ListDataset(
        [
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1),
        ]
    )

    class PlanOnlyTemporalNeighborSampler(TemporalNeighborSampler):
        def sample(self, item):
            raise AssertionError("loader should use build_plan instead of the legacy sample path")

    loader = Loader(dataset=dataset, sampler=PlanOnlyTemporalNeighborSampler(num_neighbors=[-1]), batch_size=1)

    batch = next(iter(loader))
    edge_type = EDGE_TYPE

    assert torch.equal(batch.timestamp, torch.tensor([3]))
    assert torch.equal(batch.graph.edges[edge_type].timestamp, torch.tensor([1]))

def test_temporal_neighbor_sampler_prefetch_option_materializes_features_into_record_graph():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.zeros(4, 2)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
                "edge_weight": torch.zeros(3),
            }
        },
        time_attr="timestamp",
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])),
            ("edge", EDGE_TYPE, "edge_weight"): InMemoryTensorStore(torch.tensor([10.0, 20.0, 30.0])),
        }
    )
    graph.feature_store = feature_store
    sampler = TemporalNeighborSampler(
        num_neighbors=[-1],
        node_feature_names=("x",),
        edge_feature_names=("edge_weight",),
    )

    record = sampler.sample(
        TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1)
    )

    assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(record.graph.x, torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert torch.equal(record.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([10.0]))


def test_temporal_neighbor_sampler_prefetch_option_materializes_features_into_batch_graph():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.zeros(4, 2)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
                "edge_weight": torch.zeros(3),
            }
        },
        time_attr="timestamp",
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])),
            ("edge", EDGE_TYPE, "edge_weight"): InMemoryTensorStore(torch.tensor([10.0, 20.0, 30.0])),
        }
    )
    loader = Loader(
        dataset=ListDataset(
            [
                TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1),
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=feature_store,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.timestamp, torch.tensor([3]))
    assert torch.equal(batch.graph.x, torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([10.0]))

HETERO_EDGE_TYPE = ("author", "writes", "paper")


def _hetero_graph():
    return Graph.temporal(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            HETERO_EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "timestamp": torch.tensor([1, 4, 6]),
            }
        },
        time_attr="timestamp",
    )


def test_temporal_neighbor_sampler_extracts_relation_local_hetero_history_subgraph():
    graph = _hetero_graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])

    record = sampler.sample(
        TemporalEventRecord(
            graph=graph,
            src_index=1,
            dst_index=2,
            timestamp=5,
            label=1,
            edge_type=HETERO_EDGE_TYPE,
        )
    )

    assert record.edge_type == HETERO_EDGE_TYPE
    assert torch.equal(record.graph.nodes["author"].n_id, torch.tensor([1]))
    assert torch.equal(record.graph.nodes["paper"].n_id, torch.tensor([0, 2]))
    assert torch.equal(record.graph.edges[HETERO_EDGE_TYPE].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(record.graph.edges[HETERO_EDGE_TYPE].timestamp, torch.tensor([4]))
    assert record.src_index == 0
    assert record.dst_index == 1


def test_temporal_neighbor_sampler_prefetch_option_materializes_features_into_hetero_record_graph():
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.zeros(2, 2)},
            "paper": {"x": torch.zeros(3, 2)},
        },
        edges={
            HETERO_EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "timestamp": torch.tensor([1, 4, 6]),
                "edge_weight": torch.zeros(3),
            }
        },
        time_attr="timestamp",
    )
    feature_store = FeatureStore(
        {
            ("node", "author", "x"): InMemoryTensorStore(torch.tensor([[10.0, 0.0], [20.0, 0.0]])),
            ("node", "paper", "x"): InMemoryTensorStore(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])),
            ("edge", HETERO_EDGE_TYPE, "edge_weight"): InMemoryTensorStore(torch.tensor([11.0, 22.0, 33.0])),
        }
    )
    graph.feature_store = feature_store
    sampler = TemporalNeighborSampler(
        num_neighbors=[-1],
        node_feature_names={"author": ("x",), "paper": ("x",)},
        edge_feature_names={HETERO_EDGE_TYPE: ("edge_weight",)},
    )

    record = sampler.sample(
        TemporalEventRecord(
            graph=graph,
            src_index=1,
            dst_index=2,
            timestamp=5,
            label=1,
            edge_type=HETERO_EDGE_TYPE,
        )
    )

    assert torch.equal(record.graph.nodes["author"].x, torch.tensor([[20.0, 0.0]]))
    assert torch.equal(record.graph.nodes["paper"].x, torch.tensor([[1.0, 0.0], [3.0, 0.0]]))
    assert torch.equal(record.graph.edges[HETERO_EDGE_TYPE].edge_weight, torch.tensor([22.0]))




def test_temporal_neighbor_sampler_stitched_temporal_sampling_crosses_partition_boundaries_through_coordinator(tmp_path):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                "timestamp": torch.tensor([1, 3, 5]),
                "edge_weight": torch.tensor([10.0, 20.0, 30.0]),
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
    loader = Loader(
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
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.timestamp, torch.tensor([4]))
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].timestamp, torch.tensor([1, 3]))
    assert torch.equal(batch.graph.x, torch.tensor([[0.0], [1.0], [2.0]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))


def test_temporal_neighbor_sampler_stitched_temporal_sampling_crosses_partition_boundaries_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                "timestamp": torch.tensor([1, 3, 5]),
                "edge_weight": torch.tensor([10.0, 20.0, 30.0]),
            }
        },
        time_attr="timestamp",
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
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
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.timestamp, torch.tensor([4]))
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].timestamp, torch.tensor([1, 3]))
    assert torch.equal(batch.graph.x, torch.tensor([[0.0], [1.0], [2.0]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))



def test_temporal_neighbor_sampler_prefetch_option_keeps_sampled_shard_global_ids_aligned_through_coordinator(tmp_path):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 2], [1, 3]]),
                "timestamp": torch.tensor([1, 3]),
                "edge_weight": torch.tensor([10.0, 20.0]),
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
    loader = Loader(
        dataset=ListDataset(
            [
                TemporalEventRecord(
                    graph=shards[1].graph,
                    src_index=0,
                    dst_index=1,
                    timestamp=4,
                    label=1,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([2, 3]))
    assert torch.equal(batch.graph.x, torch.tensor([[2.0], [3.0]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([20.0]))


def test_temporal_neighbor_sampler_prefetch_option_keeps_sampled_shard_global_ids_aligned_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 2], [1, 3]]),
                "timestamp": torch.tensor([1, 3]),
                "edge_weight": torch.tensor([10.0, 20.0]),
            }
        },
        time_attr="timestamp",
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                TemporalEventRecord(
                    graph=shards[1].graph,
                    src_index=0,
                    dst_index=1,
                    timestamp=4,
                    label=1,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([2, 3]))
    assert torch.equal(batch.graph.x, torch.tensor([[2.0], [3.0]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([20.0]))



def test_temporal_neighbor_sampler_stitched_hetero_temporal_sampling_crosses_partition_boundaries_through_coordinator(tmp_path):
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            HETERO_EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 2, 1], [0, 0, 1]]),
                "timestamp": torch.tensor([1, 3, 6]),
                "edge_weight": torch.tensor([10.0, 20.0, 30.0]),
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
    loader = Loader(
        dataset=ListDataset(
            [
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    timestamp=4,
                    label=1,
                    edge_type=HETERO_EDGE_TYPE,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={HETERO_EDGE_TYPE: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.edge_type == HETERO_EDGE_TYPE
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.timestamp, torch.tensor([4]))
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0]))
    assert torch.equal(batch.graph.nodes["author"].x, torch.tensor([[10.0], [30.0]]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[1.0]]))
    assert torch.equal(
        batch.graph.edges[HETERO_EDGE_TYPE].edge_index,
        torch.tensor([[0, 1], [0, 0]]),
    )
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].timestamp, torch.tensor([1, 3]))
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([0]))


def test_temporal_neighbor_sampler_stitched_hetero_temporal_sampling_crosses_partition_boundaries_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            HETERO_EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 2, 1], [0, 0, 1]]),
                "timestamp": torch.tensor([1, 3, 6]),
                "edge_weight": torch.tensor([10.0, 20.0, 30.0]),
            }
        },
        time_attr="timestamp",
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    timestamp=4,
                    label=1,
                    edge_type=HETERO_EDGE_TYPE,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={HETERO_EDGE_TYPE: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.edge_type == HETERO_EDGE_TYPE
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.timestamp, torch.tensor([4]))
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0]))
    assert torch.equal(batch.graph.nodes["author"].x, torch.tensor([[10.0], [30.0]]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[1.0]]))
    assert torch.equal(
        batch.graph.edges[HETERO_EDGE_TYPE].edge_index,
        torch.tensor([[0, 1], [0, 0]]),
    )
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].timestamp, torch.tensor([1, 3]))
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([0]))
