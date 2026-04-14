import pytest
import torch

from vgl.dataloading.requests import GraphSeedRequest, LinkSeedRequest, NodeSeedRequest, TemporalSeedRequest


EDGE_TYPE = ("paper", "cites", "paper")


def test_node_seed_request_preserves_metadata_and_validates_rank():
    request = NodeSeedRequest(
        node_ids=torch.tensor([2, 0]),
        node_type="paper",
        metadata={"split": "train"},
    )

    assert request.kind == "node"
    assert request.seed_count == 2
    assert request.node_type == "paper"
    assert request.node_ids.tolist() == [2, 0]
    assert request.seed_tensors == (request.node_ids,)
    assert request.metadata == {"split": "train"}
    assert request.resolved_sample_id is None
    assert request.resolved_query_id is None

    with pytest.raises(ValueError, match="rank-1"):
        NodeSeedRequest(node_ids=torch.tensor([[0, 1]]), node_type="paper")

    normalized = NodeSeedRequest(node_ids=[4, 3], node_type="paper")

    assert torch.equal(normalized.node_ids, torch.tensor([4, 3]))


def test_seed_requests_resolve_sample_and_query_ids_from_fields_or_metadata():
    node_request = NodeSeedRequest(
        node_ids=torch.tensor([2, 0]),
        metadata={"sample_id": "node-sample"},
    )
    assert node_request.resolved_sample_id == "node-sample"
    assert node_request.resolved_query_id == "node-sample"

    link_request = LinkSeedRequest(
        src_ids=torch.tensor([0, 1]),
        dst_ids=torch.tensor([1, 2]),
        sample_id="link-sample",
        query_id="link-query",
    )
    assert link_request.resolved_sample_id == "link-sample"
    assert link_request.resolved_query_id == "link-query"

    temporal_request = TemporalSeedRequest(
        src_ids=torch.tensor([0]),
        dst_ids=torch.tensor([1]),
        timestamps=torch.tensor([10]),
        metadata={"query_id": "temporal-query", "sample_id": "temporal-sample"},
    )
    assert temporal_request.resolved_sample_id == "temporal-sample"
    assert temporal_request.resolved_query_id == "temporal-query"

    graph_request = GraphSeedRequest(
        graph_ids=torch.tensor([3, 1]),
        sample_id="graph-sample",
    )
    assert graph_request.resolved_sample_id == "graph-sample"
    assert graph_request.resolved_query_id == "graph-sample"


def test_link_seed_request_validates_pair_lengths():
    request = LinkSeedRequest(
        src_ids=torch.tensor([0, 1]),
        dst_ids=torch.tensor([1, 2]),
        edge_type=EDGE_TYPE,
        labels=torch.tensor([1, 0]),
        metadata={"split": "valid"},
    )

    assert request.kind == "link"
    assert request.seed_count == 2
    assert request.edge_type == EDGE_TYPE
    assert request.labels.tolist() == [1, 0]
    assert request.seed_tensors == (request.src_ids, request.dst_ids)
    assert request.metadata == {"split": "valid"}

    with pytest.raises(ValueError, match="same length"):
        LinkSeedRequest(
            src_ids=torch.tensor([0]),
            dst_ids=torch.tensor([1, 2]),
            edge_type=EDGE_TYPE,
        )


def test_temporal_seed_request_tracks_timestamps():
    request = TemporalSeedRequest(
        src_ids=torch.tensor([0, 1]),
        dst_ids=torch.tensor([1, 2]),
        timestamps=torch.tensor([10, 20]),
        edge_type=EDGE_TYPE,
    )

    assert request.kind == "temporal"
    assert request.seed_count == 2
    assert request.timestamps.tolist() == [10, 20]
    assert request.seed_tensors == (request.src_ids, request.dst_ids, request.timestamps)

    with pytest.raises(ValueError, match="same length"):
        TemporalSeedRequest(
            src_ids=torch.tensor([0]),
            dst_ids=torch.tensor([1]),
            timestamps=torch.tensor([10, 20]),
            edge_type=EDGE_TYPE,
        )


def test_graph_seed_request_normalizes_graph_classification_inputs():
    request = GraphSeedRequest(
        graph_ids=torch.tensor([3, 1]),
        labels=torch.tensor([1, 0]),
        metadata={"task": "graph-classification"},
    )

    assert request.kind == "graph"
    assert request.seed_count == 2
    assert request.graph_ids.tolist() == [3, 1]
    assert request.labels.tolist() == [1, 0]
    assert request.seed_tensors == (request.graph_ids,)
    assert request.metadata == {"task": "graph-classification"}

    with pytest.raises(ValueError, match="same length"):
        GraphSeedRequest(
            graph_ids=torch.tensor([0]),
            labels=torch.tensor([1, 0]),
        )

    normalized = GraphSeedRequest(graph_ids=[3, 1], labels=[1, 0])

    assert torch.equal(normalized.graph_ids, torch.tensor([3, 1]))
    assert torch.equal(normalized.labels, torch.tensor([1, 0]))
