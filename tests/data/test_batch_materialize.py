import torch

from vgl import Graph
from vgl.core.batch import GraphBatch, LinkPredictionBatch, NodeBatch, TemporalEventBatch
from vgl.data.sample import LinkPredictionRecord, TemporalEventRecord
from vgl.dataloading.executor import MaterializationContext
from vgl.dataloading.materialize import materialize_batch
from vgl.dataloading.requests import GraphSeedRequest, LinkSeedRequest, NodeSeedRequest, TemporalSeedRequest


EDGE_TYPE = ("node", "interacts", "node")


def test_materialize_batch_builds_node_batch_from_node_contexts():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.randn(5, 4),
        y=torch.tensor([0, 1, 0, 1, 0]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([2]),
            node_type="node",
            metadata={"seed": 2, "sample_id": "n2"},
        ),
        state={"node_ids": torch.tensor([1, 2, 3])},
        metadata={"sample_id": "n2"},
        graph=graph,
    )

    batch = materialize_batch([context])

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.graph.n_id, torch.tensor([1, 2, 3]))
    assert torch.equal(batch.seed_index, torch.tensor([1]))
    assert batch.metadata == [{"seed": 2, "sample_id": "n2"}]


def test_materialize_batch_builds_graph_batch_from_graph_contexts():
    contexts = [
        MaterializationContext(
            request=GraphSeedRequest(
                graph_ids=torch.tensor([0]),
                labels=torch.tensor([1]),
                metadata={"label": 1},
            ),
            state={
                "graph": Graph.homo(
                    edge_index=torch.tensor([[0], [1]]),
                    x=torch.randn(2, 4),
                    y=torch.tensor([1]),
                )
            },
        ),
        MaterializationContext(
            request=GraphSeedRequest(
                graph_ids=torch.tensor([1]),
                labels=torch.tensor([0]),
                metadata={"label": 0},
            ),
            state={
                "graph": Graph.homo(
                    edge_index=torch.tensor([[0], [1]]),
                    x=torch.randn(2, 4),
                    y=torch.tensor([0]),
                )
            },
        ),
    ]

    batch = materialize_batch(contexts, label_source="metadata", label_key="label")

    assert isinstance(batch, GraphBatch)
    assert batch.num_graphs == 2
    assert torch.equal(batch.labels, torch.tensor([1, 0]))


def test_materialize_batch_builds_link_prediction_batch_from_link_contexts():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )
    contexts = [
        MaterializationContext(
            request=LinkSeedRequest(
                src_ids=torch.tensor([0]),
                dst_ids=torch.tensor([1]),
                labels=torch.tensor([1]),
            ),
            state={"record": LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1)},
        ),
        MaterializationContext(
            request=LinkSeedRequest(
                src_ids=torch.tensor([2]),
                dst_ids=torch.tensor([0]),
                labels=torch.tensor([0]),
            ),
            state={"record": LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0)},
        ),
    ]

    batch = materialize_batch(contexts)

    assert isinstance(batch, LinkPredictionBatch)
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))


def test_materialize_batch_builds_temporal_event_batch_from_temporal_contexts():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([1, 4]),
            }
        },
        time_attr="timestamp",
    )
    contexts = [
        MaterializationContext(
            request=TemporalSeedRequest(
                src_ids=torch.tensor([0]),
                dst_ids=torch.tensor([1]),
                timestamps=torch.tensor([1]),
                edge_type=EDGE_TYPE,
            ),
            state={"record": TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=1, label=1)},
        ),
        MaterializationContext(
            request=TemporalSeedRequest(
                src_ids=torch.tensor([1]),
                dst_ids=torch.tensor([2]),
                timestamps=torch.tensor([4]),
                edge_type=EDGE_TYPE,
            ),
            state={"record": TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=4, label=0)},
        ),
    ]

    batch = materialize_batch(contexts)

    assert isinstance(batch, TemporalEventBatch)
    assert torch.equal(batch.timestamp, torch.tensor([1, 4]))
    assert torch.equal(batch.labels, torch.tensor([1, 0]))
