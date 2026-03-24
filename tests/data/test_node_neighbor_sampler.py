import torch

from vgl import Graph
from vgl.core.batch import NodeBatch
from vgl.data.loader import Loader
from vgl.data.dataset import ListDataset
from vgl.data.sampler import NodeNeighborSampler


def test_node_neighbor_sampler_extracts_local_subgraph_and_seed_index():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
        y=torch.tensor([0, 1, 0, 1]),
    )
    sampler = NodeNeighborSampler(num_neighbors=[-1])

    sample = sampler.sample((graph, {"seed": 1, "sample_id": "n1"}))

    assert sample.sample_id == "n1"
    assert sample.subgraph_seed == 1
    assert torch.equal(sample.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(sample.graph.edge_index, torch.tensor([[0, 1, 3], [1, 2, 1]]))


def test_loader_builds_node_batch_from_node_neighbor_samples():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.randn(5, 4),
        y=torch.tensor([0, 1, 0, 1, 0]),
    )
    dataset = ListDataset(
        [
            (graph, {"seed": 0, "sample_id": "n0"}),
            (graph, {"seed": 2, "sample_id": "n2"}),
        ]
    )
    loader = Loader(dataset=dataset, sampler=NodeNeighborSampler(num_neighbors=[-1]), batch_size=2)

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.graph.x.size(0) == 5
    assert torch.equal(batch.seed_index, torch.tensor([0, 3]))
    assert batch.metadata == [{"seed": 0, "sample_id": "n0"}, {"seed": 2, "sample_id": "n2"}]


def test_node_neighbor_sampler_extracts_hetero_local_subgraph():
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.randn(3, 4),
                "y": torch.tensor([0, 1, 0]),
            },
            "author": {
                "x": torch.randn(2, 4),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 2], [0, 1]]),
            },
        },
    )
    sampler = NodeNeighborSampler(num_neighbors=[-1])

    sample = sampler.sample((graph, {"seed": 1, "node_type": "paper", "sample_id": "p1"}))

    assert sample.sample_id == "p1"
    assert sample.metadata["node_type"] == "paper"
    assert sample.subgraph_seed == 0
    assert torch.equal(sample.graph.nodes["paper"].data["n_id"], torch.tensor([1]))
    assert torch.equal(sample.graph.nodes["author"].data["n_id"], torch.tensor([0]))
    assert torch.equal(sample.graph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0], [0]]))


def test_loader_routes_node_neighbor_sampler_through_plan_execution():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.randn(5, 4),
        y=torch.tensor([0, 1, 0, 1, 0]),
    )
    dataset = ListDataset(
        [
            (graph, {"seed": 0, "sample_id": "n0"}),
            (graph, {"seed": 2, "sample_id": "n2"}),
        ]
    )

    class PlanOnlyNodeNeighborSampler(NodeNeighborSampler):
        def sample(self, item):
            raise AssertionError("loader should use build_plan instead of the legacy sample path")

    loader = Loader(dataset=dataset, sampler=PlanOnlyNodeNeighborSampler(num_neighbors=[-1]), batch_size=2)

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.graph.x.size(0) == 5
    assert torch.equal(batch.seed_index, torch.tensor([0, 3]))
    assert batch.metadata == [{"seed": 0, "sample_id": "n0"}, {"seed": 2, "sample_id": "n2"}]
