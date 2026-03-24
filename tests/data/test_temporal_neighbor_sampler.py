import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import TemporalEventRecord
from vgl.data.sampler import TemporalNeighborSampler


def _graph():
    return Graph.temporal(
        nodes={"node": {"x": torch.randn(4, 4)}},
        edges={
            ("node", "interacts", "node"): {
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
    edge_type = ("node", "interacts", "node")

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
    edge_type = ("node", "interacts", "node")

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
    edge_type = ("node", "interacts", "node")

    assert torch.equal(batch.timestamp, torch.tensor([3]))
    assert torch.equal(batch.graph.edges[edge_type].timestamp, torch.tensor([1]))
