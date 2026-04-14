import torch

from vgl import Graph
from vgl.dataloading import (
    CandidateLinkSampler,
    DataLoader,
    HardNegativeLinkSampler,
    LinkNeighborSampler,
    LinkPredictionRecord,
    ListDataset,
    NodeNeighborSampler,
    TemporalEventRecord,
    TemporalNeighborSampler,
    UniformNegativeLinkSampler,
)
from vgl.graph.batch import NodeBatch


class _ResolvedLinkPredictionRecord(LinkPredictionRecord):
    @property
    def resolved_sample_id(self):
        return "resolved-sample"

    @property
    def resolved_query_id(self):
        return "resolved-query"


def test_node_neighbor_sampler_accepts_replace_and_directed_aliases():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        x=torch.randn(3, 2),
    )
    loader = DataLoader(
        dataset=ListDataset([(graph, {"seed": 1})]),
        sampler=NodeNeighborSampler([2], replace=False, directed=False),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.seed_index.numel() == 1


def test_link_neighbor_sampler_can_read_candidate_pool_from_custom_metadata_key():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(3, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"candidate_pool": [2]},
    )
    sampler = LinkNeighborSampler(
        [1],
        base_sampler=CandidateLinkSampler(),
        candidate_dst_metadata_key="candidate_pool",
    )

    sampled = sampler.sample(record)

    assert any(int(item.dst_index) == 2 for item in sampled if int(item.label) == 0)


def test_link_neighbor_sampler_normalizes_candidate_pool_onto_metadata_without_base_sampler():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(3, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"candidate_pool": [2]},
    )
    sampler = LinkNeighborSampler(
        [1],
        candidate_dst_metadata_key="candidate_pool",
    )

    sampled = sampler.sample(record)

    assert sampled.candidate_dst == [2]
    assert sampled.metadata["candidate_dst"] == [2]


def test_link_neighbor_sampler_normalizes_field_backed_metadata_onto_localized_records():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(3, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        sample_id="edge-sample-0",
        query_id="edge-query-0",
        candidate_dst=[2],
        exclude_seed_edge=True,
    )
    sampler = LinkNeighborSampler([1])

    sampled = sampler.sample(record)

    assert sampled.metadata["sample_id"] == "edge-sample-0"
    assert sampled.metadata["query_id"] == "edge-query-0"
    assert sampled.metadata["candidate_dst"] == [2]
    assert sampled.metadata["exclude_seed_edges"] is True
    assert sampled.metadata["edge_type"] == graph._default_edge_type()


def test_candidate_link_sampler_can_read_candidate_pool_from_custom_metadata_key():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(3, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"candidate_pool": [2]},
    )
    sampler = CandidateLinkSampler(candidate_dst_metadata_key="candidate_pool")

    sampled = sampler.sample(record)

    assert sampled[0].candidate_dst == [2]
    assert all(item.candidate_dst == [2] for item in sampled)
    assert [(int(item.dst_index), int(item.label)) for item in sampled] == [(1, 1), (2, 0)]


def test_candidate_link_sampler_normalizes_resolved_candidate_pool_onto_emitted_metadata():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(3, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"candidate_pool": [2]},
    )
    sampler = CandidateLinkSampler(candidate_dst_metadata_key="candidate_pool")

    sampled = sampler.sample(record)

    assert [item.metadata["candidate_dst"] for item in sampled] == [[2], [2]]


def test_hard_negative_link_sampler_can_read_hard_negative_pool_from_custom_metadata_key():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"hard_pool": [2, 3]},
    )
    sampler = HardNegativeLinkSampler(
        num_negatives=2,
        num_hard_negatives=2,
        hard_negative_dst_metadata_key="hard_pool",
    )

    sampled = sampler.sample(record)

    assert sampled[0].hard_negative_dst == [2, 3]
    assert all(item.hard_negative_dst == [2, 3] for item in sampled)
    negatives = [item for item in sampled if int(item.label) == 0]
    assert len(negatives) == 2
    assert {int(item.dst_index) for item in negatives} == {2, 3}
    assert all(item.metadata["hard_negative_sampled"] for item in negatives)


def test_hard_negative_link_sampler_normalizes_resolved_hard_negative_pool_onto_emitted_metadata():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"hard_pool": [2, 3]},
    )
    sampler = HardNegativeLinkSampler(
        num_negatives=2,
        num_hard_negatives=2,
        hard_negative_dst_metadata_key="hard_pool",
    )

    sampled = sampler.sample(record)

    assert all(item.metadata["hard_negative_dst"] == [2, 3] for item in sampled)


def test_uniform_negative_link_sampler_normalizes_resolved_edge_types_onto_emitted_metadata():
    edge_type = ("author", "writes", "paper")
    reverse_edge_type = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 2)},
            "paper": {"x": torch.randn(3, 2)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            reverse_edge_type: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        edge_type=edge_type,
        reverse_edge_type=reverse_edge_type,
    )
    sampler = UniformNegativeLinkSampler(num_negatives=1)

    sampled = sampler.sample(record)

    assert all(item.metadata["edge_type"] == edge_type for item in sampled)
    assert all(item.metadata["reverse_edge_type"] == reverse_edge_type for item in sampled)


def test_uniform_negative_link_sampler_preserves_metadata_backed_query_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"query_id": "edge-query-0"},
    )
    sampler = UniformNegativeLinkSampler(num_negatives=2)

    sampled = sampler.sample(record)

    assert {item.query_id for item in sampled} == {"edge-query-0"}


def test_uniform_negative_link_sampler_can_fall_back_to_metadata_backed_sample_ids_for_query_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"sample_id": "edge-sample-0"},
    )
    sampler = UniformNegativeLinkSampler(num_negatives=2)

    sampled = sampler.sample(record)

    assert {item.query_id for item in sampled} == {"edge-sample-0"}


def test_uniform_negative_link_sampler_normalizes_metadata_backed_sample_ids_onto_emitted_records():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"sample_id": "edge-sample-0"},
    )
    sampler = UniformNegativeLinkSampler(num_negatives=2)

    sampled = sampler.sample(record)

    assert [item.sample_id for item in sampled] == [
        "edge-sample-0",
        "edge-sample-0:neg:0",
        "edge-sample-0:neg:1",
    ]
    assert [item.metadata["sample_id"] for item in sampled] == [
        "edge-sample-0",
        "edge-sample-0:neg:0",
        "edge-sample-0:neg:1",
    ]


def test_uniform_negative_link_sampler_normalizes_resolved_query_ids_onto_emitted_record_metadata():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"sample_id": "edge-sample-0"},
    )
    sampler = UniformNegativeLinkSampler(num_negatives=2)

    sampled = sampler.sample(record)

    assert [item.query_id for item in sampled] == ["edge-sample-0", "edge-sample-0", "edge-sample-0"]
    assert [item.metadata["query_id"] for item in sampled] == ["edge-sample-0", "edge-sample-0", "edge-sample-0"]


def test_uniform_negative_link_sampler_prefers_record_resolved_sample_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = _ResolvedLinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        sample_id=None,
        metadata={},
    )
    sampler = UniformNegativeLinkSampler(num_negatives=2)

    sampled = sampler.sample(record)

    assert [item.sample_id for item in sampled] == [
        "resolved-sample",
        "resolved-sample:neg:0",
        "resolved-sample:neg:1",
    ]
    assert [item.metadata["sample_id"] for item in sampled] == [
        "resolved-sample",
        "resolved-sample:neg:0",
        "resolved-sample:neg:1",
    ]


def test_uniform_negative_link_sampler_prefers_record_resolved_query_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = _ResolvedLinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        sample_id=None,
        metadata={},
    )
    sampler = UniformNegativeLinkSampler(num_negatives=2)

    sampled = sampler.sample(record)

    assert [item.query_id for item in sampled] == [
        "resolved-query",
        "resolved-query",
        "resolved-query",
    ]
    assert [item.metadata["query_id"] for item in sampled] == [
        "resolved-query",
        "resolved-query",
        "resolved-query",
    ]


def test_uniform_negative_link_sampler_normalizes_resolved_exclude_seed_edge_flags_onto_positive_metadata():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
    )
    sampler = UniformNegativeLinkSampler(num_negatives=2, exclude_seed_edges=True)

    sampled = sampler.sample(record)

    assert sampled[0].exclude_seed_edge is True
    assert sampled[0].metadata["exclude_seed_edges"] is True
    negatives = [item for item in sampled if int(item.label) == 0]
    assert all("exclude_seed_edges" not in item.metadata for item in negatives)


def test_uniform_negative_link_sampler_clears_exclude_seed_edge_metadata_from_generated_negatives():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(4, 2),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"exclude_seed_edges": True},
    )
    sampler = UniformNegativeLinkSampler(num_negatives=2)

    sampled = sampler.sample(record)

    assert sampled[0].metadata["exclude_seed_edges"] is True
    negatives = [item for item in sampled if int(item.label) == 0]
    assert len(negatives) == 2
    assert all("exclude_seed_edges" not in item.metadata for item in negatives)


def test_temporal_neighbor_sampler_can_include_seed_timestamp_edges():
    edge_type = ("node", "to", "node")
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 2)}},
        edges={
            edge_type: {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([1, 2]),
            }
        },
        time_attr="timestamp",
    )
    sampler = TemporalNeighborSampler([2], include_seed_timestamp=True)

    sampled = sampler.sample(
        TemporalEventRecord(
            graph=graph,
            src_index=1,
            dst_index=2,
            timestamp=2,
            label=1,
        )
    )

    assert sampled.graph.edge_index.size(1) == 2
