import torch

from vgl import Graph
from vgl.dataloading import (
    ClusterData,
    ClusterLoader,
    DataLoader,
    FullGraphSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    ListDataset,
    Node2VecWalkSampler,
    RandomWalkSampler,
    ShaDowKHopSampler,
)
from vgl.graph.batch import GraphBatch, NodeBatch


def _graph():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 0, 4],
        ]
    )
    return Graph.homo(edge_index=edge_index, x=torch.arange(15, dtype=torch.float32).view(5, 3))


def _dead_end_graph():
    edge_index = torch.tensor([[0], [1]])
    return Graph.homo(edge_index=edge_index, x=torch.arange(6, dtype=torch.float32).view(2, 3))


def test_random_walk_sampler_returns_path_sample_record():
    sampler = RandomWalkSampler(walk_length=3, seed=0)

    sample = sampler.sample((_graph(), {"seed": 0, "sample_id": "walk-0"}))

    assert sample.sample_id == "walk-0"
    assert sample.metadata["seed"] == 0
    assert sample.metadata["seed_ids"] == [0]
    assert sample.metadata["sampling_config"]["walk_length"] == 3
    assert sample.metadata["sampled_num_nodes"] == int(sample.graph.x.size(0))
    assert sample.metadata["sampled_num_edges"] == int(sample.graph.edge_index.size(1))
    assert sample.metadata["walk_starts"] == [0]
    assert [sample.metadata["sampled_node_ids"][index] for index in sample.metadata["walk_start_positions"]] == sample.metadata["walk_starts"]
    assert sample.metadata["sampled_node_ids"] == sample.metadata["walk_nodes"]
    assert sample.metadata["walk"][0] == 0
    assert sample.graph.edge_index.size(1) <= 3


def test_random_walk_sampler_avoids_tensor_tolist(monkeypatch):
    sampler = RandomWalkSampler(walk_length=3, seed=0)

    def fail_tolist(self):
        raise AssertionError("RandomWalkSampler should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sample = sampler.sample((_graph(), {"seed": 0, "sample_id": "walk-0"}))

    assert sample.sample_id == "walk-0"
    assert sample.metadata["walk"][0] == 0


def test_random_walk_sampler_supports_multiple_walks_from_one_seed():
    sampler = RandomWalkSampler(walk_length=3, num_walks=2, seed=0)

    sample = sampler.sample((_graph(), {"seed": 1, "sample_id": "walk-many"}))

    assert sample.sample_id == "walk-many"
    assert sample.metadata["seed"] == 1
    assert sample.metadata["seed_ids"] == [1, 1]
    assert sample.metadata["sampling_config"]["walk_length"] == 3
    assert sample.metadata["sampling_config"]["num_walks"] == 2
    assert "walk" not in sample.metadata
    assert len(sample.metadata["walks"]) == 2
    assert all(walk[0] == 1 for walk in sample.metadata["walks"])
    assert sample.graph.edge_index.size(1) <= 6


def test_random_walk_sampler_accepts_explicit_seed_collections():
    sampler = RandomWalkSampler(walk_length=2, seed=0)

    sample = sampler.sample((_graph(), {"seed": [0, 3]}))

    assert sample.metadata["seed"] == [0, 3]
    assert sample.metadata["seed_ids"] == [0, 3]
    assert sample.metadata["sampling_config"]["num_walks"] == 1
    assert sample.metadata["sampled_num_walks"] == 2
    assert "walk" not in sample.metadata
    assert len(sample.metadata["walks"]) == 2
    assert [walk[0] for walk in sample.metadata["walks"]] == [0, 3]


def test_random_walk_sampler_can_expand_explicit_seed_collections_into_node_samples():
    sampler = RandomWalkSampler(walk_length=2, seed=0, expand_seeds=True)

    samples = sampler.sample((_graph(), {"seed": [0, 3], "sample_id": "walk-many"}))

    assert isinstance(samples, list)
    assert len(samples) == 2
    assert samples[0].graph is samples[1].graph
    assert [sample.sample_id for sample in samples] == ["walk-many", "walk-many"]
    assert [sample.metadata["seed"] for sample in samples] == [0, 3]
    assert all(sample.metadata["seed_ids"] == [0, 3] for sample in samples)
    assert all(sample.subgraph_seed is not None for sample in samples)
    assert all("sampled_node_ids" in sample.metadata for sample in samples)


def test_random_walk_sampler_records_effective_walk_lengths():
    sampler = RandomWalkSampler(walk_length=3, seed=0)

    sample = sampler.sample((_dead_end_graph(), {"seed": 0}))

    assert sample.metadata["walk_lengths"] == [2]
    assert sample.metadata["walk_ended_early"] == [True]
    assert sample.metadata["num_walks_ended_early"] == 1
    assert sample.metadata["walk_edge_pairs"] == [[[0, 1]]]
    assert sample.metadata["walk"] == [0, 1, -1, -1]


def test_node2vec_walk_sampler_returns_fixed_length_walk():
    sampler = Node2VecWalkSampler(walk_length=4, p=0.5, q=2.0, seed=1)

    sample = sampler.sample((_graph(), {"seed": 2}))

    assert sample.metadata["walk"][0] == 2
    assert sample.metadata["seed_ids"] == [2]
    assert sample.metadata["sampling_config"]["p"] == 0.5
    assert sample.metadata["sampling_config"]["q"] == 2.0
    assert sample.metadata["walk_starts"] == [2]
    assert sample.metadata["sampled_node_ids"] == sample.metadata["walk_nodes"]
    assert len(sample.metadata["walk"]) == 5


def test_node2vec_walk_sampler_avoids_tensor_tolist(monkeypatch):
    sampler = Node2VecWalkSampler(walk_length=4, p=0.5, q=2.0, seed=1)

    def fail_tolist(self):
        raise AssertionError("Node2VecWalkSampler should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sample = sampler.sample((_graph(), {"seed": 2}))

    assert sample.metadata["walk"][0] == 2
    assert len(sample.metadata["walk"]) == 5


def test_node2vec_walk_sampler_can_expand_explicit_seed_collections_into_node_samples():
    sampler = Node2VecWalkSampler(walk_length=3, p=0.5, q=2.0, seed=1, expand_seeds=True)

    samples = sampler.sample((_graph(), {"seed": [1, 3], "sample_id": "node2vec-many"}))

    assert isinstance(samples, list)
    assert len(samples) == 2
    assert samples[0].graph is samples[1].graph
    assert [sample.sample_id for sample in samples] == ["node2vec-many", "node2vec-many"]
    assert [sample.metadata["seed"] for sample in samples] == [1, 3]
    assert all(sample.metadata["seed_ids"] == [1, 3] for sample in samples)
    assert all(sample.subgraph_seed is not None for sample in samples)
    assert all("sampled_node_ids" in sample.metadata for sample in samples)


def test_node2vec_walk_sampler_supports_multiple_walks_from_one_seed():
    sampler = Node2VecWalkSampler(walk_length=3, num_walks=2, p=0.5, q=2.0, seed=1)

    sample = sampler.sample((_graph(), {"seed": 2}))

    assert sample.metadata["seed"] == 2
    assert sample.metadata["seed_ids"] == [2, 2]
    assert sample.metadata["sampling_config"]["num_walks"] == 2
    assert sample.metadata["sampled_num_walks"] == 2
    assert sample.metadata["sampling_config"]["p"] == 0.5
    assert sample.metadata["sampling_config"]["q"] == 2.0
    assert "walk" not in sample.metadata
    assert len(sample.metadata["walks"]) == 2
    assert sample.metadata["walk_starts"] == [2, 2]
    assert [sample.metadata["sampled_node_ids"][index] for index in sample.metadata["walk_start_positions"]] == sample.metadata["walk_starts"]
    assert sample.metadata["sampled_node_ids"] == sample.metadata["walk_nodes"]
    assert all(len(walk) == 4 for walk in sample.metadata["walks"])
    assert all(walk[0] == 2 for walk in sample.metadata["walks"])


def test_node2vec_walk_sampler_records_effective_walk_lengths():
    sampler = Node2VecWalkSampler(walk_length=3, p=0.5, q=2.0, seed=1)

    sample = sampler.sample((_dead_end_graph(), {"seed": 0}))

    assert sample.metadata["walk_lengths"] == [2]
    assert sample.metadata["walk_ended_early"] == [True]
    assert sample.metadata["num_walks_ended_early"] == 1
    assert sample.metadata["walk_edge_pairs"] == [[[0, 1]]]
    assert sample.metadata["walk"] == [0, 1, -1, -1]


def test_loader_batches_walk_samples_and_preserves_metadata():
    dataset = ListDataset(
        [
            (_graph(), {"seed": 0, "sample_id": "w0"}),
            (_graph(), {"seed": 1, "sample_id": "w1"}),
        ]
    )
    loader = DataLoader(dataset=dataset, sampler=RandomWalkSampler(walk_length=2, seed=2), batch_size=2)

    batch = next(iter(loader))

    assert isinstance(batch, GraphBatch)
    assert batch.metadata is not None
    assert [entry["sample_id"] for entry in batch.metadata] == ["w0", "w1"]
    assert all("walk" in entry for entry in batch.metadata)


def test_loader_builds_node_batch_from_random_walk_multi_seed_context():
    dataset = ListDataset([(_graph(), {"seed": [0, 3], "sample_id": "walk-many"})])
    loader = DataLoader(dataset=dataset, sampler=RandomWalkSampler(walk_length=2, seed=2, expand_seeds=True), batch_size=1)

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.seed_index.numel() == 2
    assert [entry["seed"] for entry in batch.metadata] == [0, 3]


def test_graph_saint_node_sampler_returns_induced_subgraph():
    sampler = GraphSAINTNodeSampler(num_sampled_nodes=3, seed=3)

    sample = sampler.sample(_graph())

    assert sample.graph.x.size(0) <= 3
    assert "sampled_node_ids" in sample.metadata
    assert "sampled_edge_ids" in sample.metadata
    assert "subgraph_edge_ids" in sample.metadata
    assert sample.metadata["sampled_num_nodes"] == int(sample.graph.x.size(0))
    assert sample.metadata["sampled_num_edges"] == int(sample.graph.edge_index.size(1))
    assert sample.metadata["sampled_edge_ids"] == sample.metadata["subgraph_edge_ids"]
    assert sample.metadata["sampled_num_edges"] == len(sample.metadata["subgraph_edge_ids"])
    assert sample.metadata["sampling_config"]["num_sampled_nodes"] == 3


def test_graph_saint_node_sampler_avoids_tensor_tolist(monkeypatch):
    sampler = GraphSAINTNodeSampler(num_sampled_nodes=3, seed=3)

    def fail_tolist(self):
        raise AssertionError("GraphSAINTNodeSampler should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sample = sampler.sample(_graph())

    assert sample.graph.x.size(0) <= 3
    assert "sampled_node_ids" in sample.metadata


def test_graph_saint_node_sampler_expands_explicit_seed_collections():
    sampler = GraphSAINTNodeSampler(num_sampled_nodes=2, seed=3)

    samples = sampler.sample((_graph(), {"seed": [1, 4], "sample_id": "saint-node-many"}))

    assert isinstance(samples, list)
    assert len(samples) == 2
    assert samples[0].graph is samples[1].graph
    assert [sample.sample_id for sample in samples] == ["saint-node-many", "saint-node-many"]
    assert [sample.metadata["seed"] for sample in samples] == [1, 4]
    assert all(sample.metadata["seed_ids"] == [1, 4] for sample in samples)
    assert [samples[0].metadata["sampled_node_ids"][index] for index in samples[0].metadata["seed_positions"]] == [1, 4]
    assert {1, 4} <= set(samples[0].metadata["sampled_node_ids"])
    assert [sample.subgraph_seed for sample in samples] == [0, 1]


def test_graph_saint_edge_sampler_returns_endpoint_induced_subgraph():
    sampler = GraphSAINTEdgeSampler(num_sampled_edges=2, seed=4)

    sample = sampler.sample(_graph())

    assert sample.graph.edge_index.size(1) >= 1
    assert "sampled_edge_ids" in sample.metadata
    assert "subgraph_edge_ids" in sample.metadata
    assert set(sample.metadata["sampled_edge_ids"]) <= set(sample.metadata["subgraph_edge_ids"])
    assert sample.metadata["sampled_num_nodes"] == int(sample.graph.x.size(0))
    assert sample.metadata["sampled_num_edges"] == int(sample.graph.edge_index.size(1))
    assert sample.metadata["sampling_config"]["num_sampled_edges"] == 2


def test_graph_saint_edge_sampler_avoids_tensor_tolist(monkeypatch):
    sampler = GraphSAINTEdgeSampler(num_sampled_edges=2, seed=4)

    def fail_tolist(self):
        raise AssertionError("GraphSAINTEdgeSampler should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sample = sampler.sample(_graph())

    assert sample.graph.edge_index.size(1) >= 1
    assert "sampled_edge_ids" in sample.metadata


def test_graph_saint_edge_sampler_can_force_include_explicit_edge_ids():
    sampler = GraphSAINTEdgeSampler(num_sampled_edges=1, seed=4)

    sample = sampler.sample((_graph(), {"edge_id": [2, 3]}))

    assert sample.metadata["sampled_edge_ids"][:2] == [2, 3]
    assert sample.metadata["sampled_node_ids"] == [1, 2]
    assert sample.metadata["seed_ids"] == [1, 2]
    assert [sample.metadata["sampled_node_ids"][index] for index in sample.metadata["seed_positions"]] == [1, 2]


def test_graph_saint_random_walk_sampler_returns_subgraph():
    sampler = GraphSAINTRandomWalkSampler(num_walks=2, walk_length=2, seed=5)

    sample = sampler.sample(_graph())

    assert sample.graph.x.size(0) >= 1
    assert "walk_nodes" in sample.metadata
    assert sample.metadata["sampled_node_ids"] == sample.metadata["walk_nodes"]
    assert "subgraph_edge_ids" in sample.metadata
    assert sample.metadata["sampled_num_nodes"] == int(sample.graph.x.size(0))
    assert sample.metadata["sampled_num_edges"] == int(sample.graph.edge_index.size(1))
    assert sample.metadata["sampling_config"]["num_walks"] == 2


def test_graph_saint_random_walk_sampler_avoids_tensor_tolist(monkeypatch):
    sampler = GraphSAINTRandomWalkSampler(num_walks=2, walk_length=2, seed=5)

    def fail_tolist(self):
        raise AssertionError("GraphSAINTRandomWalkSampler should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sample = sampler.sample(_graph())

    assert sample.graph.x.size(0) >= 1
    assert "walks" in sample.metadata


def test_graph_saint_random_walk_sampler_accepts_explicit_start_nodes():
    sampler = GraphSAINTRandomWalkSampler(num_walks=3, walk_length=2, seed=5)

    samples = sampler.sample((_graph(), {"seed": [1, 3], "sample_id": "saint-walk-many"}))

    assert isinstance(samples, list)
    assert len(samples) == 2
    assert samples[0].graph is samples[1].graph
    assert [sample.sample_id for sample in samples] == ["saint-walk-many", "saint-walk-many"]
    assert [sample.metadata["seed"] for sample in samples] == [1, 3]
    assert all(sample.metadata["seed_ids"] == [1, 3] for sample in samples)
    assert all(sample.metadata["walk_starts"] == [1, 3] for sample in samples)
    assert all(
        [sample.metadata["sampled_node_ids"][index] for index in sample.metadata["walk_start_positions"]] == sample.metadata["walk_starts"]
        for sample in samples
    )
    assert all(sample.metadata["sampled_num_walks"] == 2 for sample in samples)
    assert all(len(sample.metadata["walks"]) == 2 for sample in samples)
    assert [walk[0] for walk in samples[0].metadata["walks"]] == [1, 3]
    assert all(sample.subgraph_seed is not None for sample in samples)


def test_graph_saint_random_walk_sampler_records_effective_walk_lengths():
    sampler = GraphSAINTRandomWalkSampler(num_walks=2, walk_length=3, seed=5)

    sample = sampler.sample((_dead_end_graph(), {"seed": 0}))

    assert sample.metadata["walk_lengths"] == [2, 2]
    assert sample.metadata["walk_ended_early"] == [True, True]
    assert sample.metadata["num_walks_ended_early"] == 2
    assert sample.metadata["walk_edge_pairs"] == [[[0, 1]], [[0, 1]]]
    assert sample.metadata["walks"] == [[0, 1, -1, -1], [0, 1, -1, -1]]


def test_loader_builds_node_batch_from_graph_saint_multi_seed_context():
    dataset = ListDataset([(_graph(), {"seed": [1, 4], "sample_id": "saint-node-many"})])
    loader = DataLoader(dataset=dataset, sampler=GraphSAINTNodeSampler(num_sampled_nodes=2, seed=3), batch_size=1)

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.seed_index, torch.tensor([0, 1]))
    assert [entry["seed"] for entry in batch.metadata] == [1, 4]


def test_cluster_data_and_loader_batch_clusters():
    cluster_data = ClusterData(_graph(), num_parts=2, seed=6)
    loader = ClusterLoader(cluster_data, batch_size=2)

    batch = next(iter(loader))

    assert len(cluster_data) == 2
    assert isinstance(batch, GraphBatch)
    assert batch.metadata is not None
    assert {entry["cluster_id"] for entry in batch.metadata} == {0, 1}
    assert {entry["partition_id"] for entry in batch.metadata} == {0, 1}
    assert all(entry["num_parts"] == 2 for entry in batch.metadata)
    assert all(entry["sampled_node_ids"] == entry["node_ids"] for entry in batch.metadata)
    assert all("sampled_edge_ids" in entry for entry in batch.metadata)
    assert all("subgraph_edge_ids" in entry for entry in batch.metadata)
    assert all(entry["sampled_edge_ids"] == entry["subgraph_edge_ids"] for entry in batch.metadata)
    assert all(entry["sampled_num_nodes"] == len(entry["sampled_node_ids"]) for entry in batch.metadata)
    assert all(entry["sampled_num_edges"] == len(entry["subgraph_edge_ids"]) for entry in batch.metadata)
    assert all(entry["sampling_config"]["num_parts"] == 2 for entry in batch.metadata)


def test_cluster_data_avoids_tensor_tolist(monkeypatch):
    def fail_tolist(self):
        raise AssertionError("ClusterData should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    cluster_data = ClusterData(_graph(), num_parts=2, seed=6)

    assert len(cluster_data) == 2
    assert all("sampled_node_ids" in sample.metadata for sample in cluster_data.samples)


def test_cluster_data_avoids_torch_isin(monkeypatch):
    def fail_isin(*args, **kwargs):
        raise AssertionError("ClusterData should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    cluster_data = ClusterData(_graph(), num_parts=2, seed=6)

    assert len(cluster_data) == 2
    assert all("sampled_edge_ids" in sample.metadata for sample in cluster_data.samples)


def test_shadow_khop_sampler_materializes_node_batch():
    dataset = ListDataset([(_graph(), {"seed": 2, "sample_id": "shadow-0"})])
    loader = DataLoader(dataset=dataset, sampler=ShaDowKHopSampler(num_hops=2), batch_size=1)

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.seed_index.numel() == 1
    assert batch.metadata is not None
    assert batch.metadata[0]["seed"] == 2
    assert batch.metadata[0]["sample_id"] == "shadow-0"
    assert [batch.metadata[0]["sampled_node_ids"][index] for index in batch.metadata[0]["seed_positions"]] == [2]
    assert 2 in batch.metadata[0]["sampled_node_ids"]
    assert "subgraph_edge_ids" in batch.metadata[0]
    assert batch.metadata[0]["sampled_num_nodes"] == len(batch.metadata[0]["sampled_node_ids"])
    assert batch.metadata[0]["sampled_num_edges"] == len(batch.metadata[0]["subgraph_edge_ids"])
    assert batch.metadata[0]["sampling_config"]["num_hops"] == 2


def test_shadow_khop_sampler_avoids_tensor_tolist(monkeypatch):
    sampler = ShaDowKHopSampler(num_hops=1)

    def fail_tolist(self):
        raise AssertionError("ShaDowKHopSampler should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sample = sampler.sample((_graph(), {"seed": 2, "sample_id": "shadow-0"}))

    assert sample.sample_id == "shadow-0"
    assert 2 in sample.metadata["sampled_node_ids"]


def test_shadow_khop_sampler_expands_explicit_multi_seed_records():
    sampler = ShaDowKHopSampler(num_hops=1)

    samples = sampler.sample((_graph(), {"seed": [1, 3], "sample_id": "shadow-many"}))

    assert isinstance(samples, list)
    assert len(samples) == 2
    assert samples[0].graph is samples[1].graph
    assert [sample.sample_id for sample in samples] == ["shadow-many", "shadow-many"]
    assert [sample.metadata["seed"] for sample in samples] == [1, 3]
    assert all(sample.metadata["seed_ids"] == [1, 3] for sample in samples)
    assert all(sample.subgraph_seed is not None for sample in samples)


def test_loader_builds_node_batch_from_shadow_multi_seed_context():
    dataset = ListDataset([(_graph(), {"seed": [1, 3], "sample_id": "shadow-many"})])
    loader = DataLoader(dataset=dataset, sampler=ShaDowKHopSampler(num_hops=1), batch_size=1)

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.seed_index.numel() == 2
    assert [entry["seed"] for entry in batch.metadata] == [1, 3]


def test_cluster_loader_accepts_label_routed_graph_batches():
    cluster_data = ClusterData(_graph(), num_parts=2, seed=7)
    for sample in cluster_data.samples:
        sample.metadata["label"] = 1
    loader = ClusterLoader(cluster_data, batch_size=2, label_source="metadata", label_key="label")

    batch = next(iter(loader))

    assert isinstance(batch, GraphBatch)
    assert torch.equal(batch.labels, torch.tensor([1, 1]))
