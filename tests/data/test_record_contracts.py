import torch

from vgl import Graph
from vgl.dataloading.records import LinkPredictionRecord, SampleRecord, TemporalEventRecord


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )


def test_sample_record_exposes_node_style_contract_fields():
    record = SampleRecord(
        graph=_graph(),
        metadata={"sample_id": "meta-sample", "query_id": "meta-query"},
        subgraph_seed=1,
    )

    assert record.kind == "node"
    assert record.seed_count == 1
    assert record.resolved_sample_id == "meta-sample"
    assert record.resolved_query_id == "meta-query"


def test_link_prediction_record_resolves_sample_and_query_ids_consistently():
    record = LinkPredictionRecord(
        graph=_graph(),
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"sample_id": "meta-sample", "query_id": "meta-query"},
    )

    assert record.kind == "link"
    assert record.seed_count == 1
    assert record.resolved_sample_id == "meta-sample"
    assert record.resolved_query_id == "meta-query"

    explicit = LinkPredictionRecord(
        graph=_graph(),
        src_index=0,
        dst_index=1,
        label=1,
        sample_id="explicit-sample",
        query_id="explicit-query",
        metadata={"sample_id": "meta-sample", "query_id": "meta-query"},
    )

    assert explicit.resolved_sample_id == "explicit-sample"
    assert explicit.resolved_query_id == "explicit-query"


def test_link_prediction_record_can_fall_back_from_query_to_sample_id():
    record = LinkPredictionRecord(
        graph=_graph(),
        src_index=0,
        dst_index=1,
        label=1,
        metadata={"sample_id": "meta-sample"},
    )

    assert record.resolved_query_id == "meta-sample"


def test_temporal_event_record_exposes_temporal_contract_fields():
    record = TemporalEventRecord(
        graph=_graph(),
        src_index=0,
        dst_index=1,
        timestamp=3,
        label=1,
        metadata={"sample_id": "temporal-sample", "query_id": "temporal-query"},
    )

    assert record.kind == "temporal"
    assert record.seed_count == 1
    assert record.resolved_sample_id == "temporal-sample"
    assert record.resolved_query_id == "temporal-query"


def test_sample_and_temporal_records_can_fall_back_from_query_to_sample_id():
    sample = SampleRecord(graph=_graph(), metadata={"sample_id": "sample-only"})
    temporal = TemporalEventRecord(
        graph=_graph(),
        src_index=0,
        dst_index=1,
        timestamp=3,
        label=1,
        metadata={"sample_id": "temporal-only"},
    )

    assert sample.resolved_query_id == "sample-only"
    assert temporal.resolved_query_id == "temporal-only"
