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
    assert request.node_type == "paper"
    assert request.node_ids.tolist() == [2, 0]
    assert request.metadata == {"split": "train"}

    with pytest.raises(ValueError, match="rank-1"):
        NodeSeedRequest(node_ids=torch.tensor([[0, 1]]), node_type="paper")


def test_link_seed_request_validates_pair_lengths():
    request = LinkSeedRequest(
        src_ids=torch.tensor([0, 1]),
        dst_ids=torch.tensor([1, 2]),
        edge_type=EDGE_TYPE,
        labels=torch.tensor([1, 0]),
        metadata={"split": "valid"},
    )

    assert request.kind == "link"
    assert request.edge_type == EDGE_TYPE
    assert request.labels.tolist() == [1, 0]
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
    assert request.timestamps.tolist() == [10, 20]

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
    assert request.graph_ids.tolist() == [3, 1]
    assert request.labels.tolist() == [1, 0]
    assert request.metadata == {"task": "graph-classification"}

    with pytest.raises(ValueError, match="same length"):
        GraphSeedRequest(
            graph_ids=torch.tensor([0]),
            labels=torch.tensor([1, 0]),
        )
