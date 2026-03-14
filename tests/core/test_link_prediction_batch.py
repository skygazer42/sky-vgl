import pytest
import torch

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch
from vgl.data.sample import LinkPredictionRecord


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )


def test_link_prediction_batch_tracks_fields():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, sample_id="p"),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, sample_id="n"),
        ]
    )

    assert batch.graph is graph
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))
    assert batch.metadata == [{}, {}]


def test_link_prediction_batch_rejects_empty_records():
    with pytest.raises(ValueError, match="at least one record"):
        LinkPredictionBatch.from_records([])


def test_link_prediction_batch_rejects_mixed_graphs():
    with pytest.raises(ValueError, match="single source graph"):
        LinkPredictionBatch.from_records(
            [
                LinkPredictionRecord(graph=_graph(), src_index=0, dst_index=1, label=1),
                LinkPredictionRecord(graph=_graph(), src_index=1, dst_index=2, label=0),
            ]
        )


def test_link_prediction_batch_rejects_out_of_range_indices():
    graph = _graph()

    with pytest.raises(ValueError, match="node range"):
        LinkPredictionBatch.from_records(
            [LinkPredictionRecord(graph=graph, src_index=3, dst_index=1, label=1)]
        )


def test_link_prediction_batch_rejects_non_binary_labels():
    graph = _graph()

    with pytest.raises(ValueError, match="binary 0/1"):
        LinkPredictionBatch.from_records(
            [LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=2)]
        )
