import torch

from gnn import Graph
from gnn.data.sample import SampleRecord
from gnn.data.sampler import NodeSeedSubgraphSampler


def test_node_seed_subgraph_sampler_returns_sample_record():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
        y=torch.tensor([1]),
    )
    sampler = NodeSeedSubgraphSampler()

    sample = sampler.sample((graph, {"seed": 1, "label": 1, "sample_id": "s1"}))

    assert isinstance(sample, SampleRecord)
    assert sample.sample_id == "s1"
    assert sample.metadata["label"] == 1
