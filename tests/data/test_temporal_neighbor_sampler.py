import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import TemporalEventRecord
from vgl.data.sampler import TemporalNeighborSampler
from vgl.storage import FeatureStore, InMemoryTensorStore


EDGE_TYPE = ("node", "interacts", "node")


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

