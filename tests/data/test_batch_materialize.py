import torch
import inspect
from types import SimpleNamespace

from vgl import Graph
from vgl.core.batch import GraphBatch, LinkPredictionBatch, NodeBatch, TemporalEventBatch
from vgl.graph.schema import GraphSchema
from vgl.data.sample import LinkPredictionRecord, SampleRecord, TemporalEventRecord
import vgl.dataloading.materialize as materialize_module
from vgl.dataloading.executor import (
    MaterializationContext,
    _build_stitched_hetero_node_samples,
    _build_stitched_node_samples,
)
from vgl.dataloading.materialize import _link_message_passing_graph, materialize_batch, materialize_context
from vgl.dataloading.requests import GraphSeedRequest, LinkSeedRequest, NodeSeedRequest, TemporalSeedRequest
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore


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


def test_materialize_batch_preserves_graph_context_metadata_without_labels():
    context = MaterializationContext(
        request=GraphSeedRequest(
            graph_ids=torch.tensor([0]),
            metadata={},
        ),
        state={
            "graph": Graph.homo(
                edge_index=torch.tensor([[0], [1]]),
                x=torch.randn(2, 4),
                y=torch.tensor([1]),
            )
        },
        metadata={"sample_id": "g0", "source_graph_id": "root-0"},
    )

    batch = materialize_batch([context])

    assert isinstance(batch, GraphBatch)
    assert batch.metadata == [{"sample_id": "g0", "source_graph_id": "root-0"}]


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


def test_materialize_batch_reuses_graph_materialization_across_link_contexts(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.zeros(3, 1),
    )
    contexts = [
        MaterializationContext(
            request=LinkSeedRequest(
                src_ids=torch.tensor([0]),
                dst_ids=torch.tensor([1]),
                labels=torch.tensor([1]),
            ),
            state={
                "record": LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
                "_materialized_node_features": {
                    "node": {
                        "x": SimpleNamespace(
                            index=torch.tensor([0, 1, 2]),
                            values=torch.tensor([[1.0], [2.0], [3.0]]),
                        )
                    }
                },
            },
        ),
        MaterializationContext(
            request=LinkSeedRequest(
                src_ids=torch.tensor([2]),
                dst_ids=torch.tensor([0]),
                labels=torch.tensor([0]),
            ),
            state={
                "record": LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
                "_materialized_node_features": {
                    "node": {
                        "x": SimpleNamespace(
                            index=torch.tensor([0, 1, 2]).clone(),
                            values=torch.tensor([[1.0], [2.0], [3.0]]).clone(),
                        )
                    }
                },
            },
        ),
    ]
    real_materialize_graph = materialize_module._graph_with_materialized_features
    calls = 0

    def counting_materialize_graph(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_materialize_graph(*args, **kwargs)

    monkeypatch.setattr(materialize_module, "_graph_with_materialized_features", counting_materialize_graph)

    batch = materialize_batch(contexts)

    assert isinstance(batch, LinkPredictionBatch)
    assert calls == 1
    assert torch.equal(batch.graph.x, torch.tensor([[1.0], [2.0], [3.0]]))


def test_materialize_batch_link_context_homo_graph_without_x_uses_fetched_node_features():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, 0]),
    )
    context = MaterializationContext(
        request=LinkSeedRequest(
            src_ids=torch.tensor([0]),
            dst_ids=torch.tensor([1]),
            labels=torch.tensor([1]),
        ),
        state={
            "record": LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            "_materialized_node_features": {
                "node": {
                    "x": SimpleNamespace(
                        index=torch.tensor([0, 1, 2]),
                        values=torch.tensor([[1.0], [2.0], [3.0]]),
                    )
                }
            },
        },
    )

    batch = materialize_batch([context])

    assert isinstance(batch, LinkPredictionBatch)
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.graph.x, torch.tensor([[1.0], [2.0], [3.0]]))


def test_materialize_batch_link_context_homo_graph_without_x_still_batches():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, 0]),
    )
    context = MaterializationContext(
        request=LinkSeedRequest(
            src_ids=torch.tensor([0]),
            dst_ids=torch.tensor([1]),
            labels=torch.tensor([1]),
        ),
        state={"record": LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1)},
    )

    batch = materialize_batch([context])

    assert isinstance(batch, LinkPredictionBatch)
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))


def test_materialize_batch_reuses_link_message_passing_graph_across_block_contexts(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.arange(3, dtype=torch.float32).view(3, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0])},
    )
    contexts = [
        MaterializationContext(
            request=LinkSeedRequest(
                src_ids=torch.tensor([0]),
                dst_ids=torch.tensor([1]),
                labels=torch.tensor([1]),
            ),
            state={
                "record": LinkPredictionRecord(
                    graph=graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    metadata={"exclude_seed_edges": True},
                ),
                "link_node_ids_local": torch.tensor([0, 1, 2]),
                "link_node_hops": [torch.tensor([0, 1]), torch.tensor([0, 1, 2])],
            },
        ),
        MaterializationContext(
            request=LinkSeedRequest(
                src_ids=torch.tensor([1]),
                dst_ids=torch.tensor([1]),
                labels=torch.tensor([1]),
            ),
            state={
                "record": LinkPredictionRecord(
                    graph=graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    metadata={"exclude_seed_edges": True},
                ),
                "link_node_ids_local": torch.tensor([0, 1, 2]).clone(),
                "link_node_hops": [torch.tensor([0, 1]).clone(), torch.tensor([0, 1, 2]).clone()],
            },
        ),
    ]
    real_message_passing_graph = materialize_module._link_message_passing_graph
    calls = 0

    def counting_message_passing_graph(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_message_passing_graph(*args, **kwargs)

    monkeypatch.setattr(materialize_module, "_link_message_passing_graph", counting_message_passing_graph)

    batch = materialize_batch(contexts)

    assert isinstance(batch, LinkPredictionBatch)
    assert calls == 1
    assert batch.blocks is not None
    assert len(batch.blocks) == 1


def test_materialize_batch_reuses_homo_subgraph_across_node_contexts(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.arange(3, dtype=torch.float32).view(3, 1),
        y=torch.tensor([0, 1, 0]),
    )
    contexts = [
        MaterializationContext(
            request=NodeSeedRequest(
                node_ids=torch.tensor([0]),
                node_type="node",
                metadata={"seed": 0, "sample_id": "n0"},
            ),
            state={"node_ids": torch.tensor([0, 1, 2])},
            metadata={"sample_id": "n0"},
            graph=graph,
        ),
        MaterializationContext(
            request=NodeSeedRequest(
                node_ids=torch.tensor([1]),
                node_type="node",
                metadata={"seed": 1, "sample_id": "n1"},
            ),
            state={"node_ids": torch.tensor([0, 1, 2]).clone()},
            metadata={"sample_id": "n1"},
            graph=graph,
        ),
    ]
    real_subgraph = materialize_module._subgraph
    calls = 0

    def counting_subgraph(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_subgraph(*args, **kwargs)

    monkeypatch.setattr(materialize_module, "_subgraph", counting_subgraph)

    batch = materialize_batch(contexts)

    assert isinstance(batch, NodeBatch)
    assert calls == 1
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.seed_index, torch.tensor([0, 1]))


def test_materialize_batch_reuses_homo_node_blocks_across_node_contexts(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.arange(3, dtype=torch.float32).view(3, 1),
        y=torch.tensor([0, 1, 0]),
    )
    contexts = [
        MaterializationContext(
            request=NodeSeedRequest(
                node_ids=torch.tensor([0]),
                node_type="node",
                metadata={"seed": 0, "sample_id": "n0"},
            ),
            state={
                "node_ids": torch.tensor([0, 1, 2]),
                "node_hops": [torch.tensor([0, 1]), torch.tensor([0, 1, 2])],
            },
            metadata={"sample_id": "n0"},
            graph=graph,
        ),
        MaterializationContext(
            request=NodeSeedRequest(
                node_ids=torch.tensor([1]),
                node_type="node",
                metadata={"seed": 1, "sample_id": "n1"},
            ),
            state={
                "node_ids": torch.tensor([0, 1, 2]).clone(),
                "node_hops": [torch.tensor([0, 1]).clone(), torch.tensor([0, 1, 2]).clone()],
            },
            metadata={"sample_id": "n1"},
            graph=graph,
        ),
    ]
    real_build_blocks = materialize_module._build_homo_blocks_from_local_ids
    calls = 0

    def counting_build_blocks(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_build_blocks(*args, **kwargs)

    monkeypatch.setattr(materialize_module, "_build_homo_blocks_from_local_ids", counting_build_blocks)

    batch = materialize_batch(contexts)

    assert isinstance(batch, NodeBatch)
    assert calls == 1
    assert batch.blocks is not None
    assert len(batch.blocks) == 1


def test_materialize_context_homo_graph_without_x_builds_node_blocks():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        y=torch.tensor([0, 1, 0]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([0]),
            node_type="node",
            metadata={"seed": 0, "sample_id": "n0"},
        ),
        state={
            "sample": SampleRecord(
                graph=graph,
                metadata={"seed": 0, "sample_id": "n0"},
                subgraph_seed=0,
            ),
            "node_hops": [torch.tensor([0, 1]), torch.tensor([0, 1, 2])],
        },
        metadata={"sample_id": "n0"},
        graph=graph,
    )

    sample = materialize_context(context)

    assert sample.blocks is not None
    assert len(sample.blocks) == 1
    assert sample.subgraph_seed == 0


def test_materialize_batch_builds_node_batch_from_multi_seed_node_contexts():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 4], [2, 4, 0]]),
        x=torch.randn(5, 4),
        y=torch.tensor([0, 1, 0, 1, 0]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([2, 4]),
            node_type="node",
            metadata={"seed": [2, 4], "sample_id": "n24"},
        ),
        state={"node_ids": torch.tensor([0, 2, 4])},
        metadata={"sample_id": "n24"},
        graph=graph,
    )

    batch = materialize_batch([context])

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 2, 4]))
    assert torch.equal(batch.seed_index, torch.tensor([1, 2]))
    assert batch.metadata == [
        {"seed": 2, "sample_id": "n24"},
        {"seed": 4, "sample_id": "n24"},
    ]


def test_materialize_batch_builds_hetero_node_batch_from_multi_seed_node_contexts():
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4), "y": torch.tensor([0, 1, 0])},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            written_by: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1, 2]),
            node_type="paper",
            metadata={"seed": [1, 2], "node_type": "paper", "sample_id": "p12"},
        ),
        state={
            "node_ids_by_type": {
                "paper": torch.tensor([1, 2]),
                "author": torch.tensor([0, 1]),
            }
        },
        metadata={"sample_id": "p12"},
        graph=graph,
    )

    batch = materialize_batch([context])

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([1, 2]))
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 1]))
    assert torch.equal(batch.seed_index, torch.tensor([0, 1]))
    assert batch.metadata == [
        {"seed": 1, "node_type": "paper", "sample_id": "p12"},
        {"seed": 2, "node_type": "paper", "sample_id": "p12"},
    ]


def test_materialize_multi_seed_homo_node_context_avoids_tensor_item(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 4], [2, 4, 0]]),
        x=torch.randn(5, 4),
        y=torch.tensor([0, 1, 0, 1, 0]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([2, 4]),
            node_type="node",
            metadata={"seed": [2, 4], "sample_id": "n24"},
        ),
        state={"node_ids": torch.tensor([0, 2, 4])},
        metadata={"sample_id": "n24"},
        graph=graph,
    )

    def fail_item(self):
        raise AssertionError("homo node materialization should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    samples = materialize_context(context)

    assert [sample.metadata["seed"] for sample in samples] == [2, 4]
    assert [sample.subgraph_seed for sample in samples] == [1, 2]


def test_materialize_multi_seed_homo_node_context_avoids_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 4], [2, 4, 0]]),
        x=torch.randn(5, 4),
        y=torch.tensor([0, 1, 0, 1, 0]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([2, 4]),
            node_type="node",
            metadata={"seed": [2, 4], "sample_id": "n24"},
        ),
        state={"node_ids": torch.tensor([0, 2, 4])},
        metadata={"sample_id": "n24"},
        graph=graph,
    )

    def fail_int(self):
        raise AssertionError("homo node materialization should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    samples = materialize_context(context)

    assert [sample.metadata["seed"] for sample in samples] == [2, 4]
    assert [sample.subgraph_seed for sample in samples] == [1, 2]


def test_materialize_context_preserves_public_seed_metadata_for_homo_subgraphs():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        n_id=torch.tensor([10, 11, 12]),
        edge_data={"e_id": torch.tensor([20, 21, 22])},
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1]),
            node_type="node",
            metadata={"seed": 11, "sample_id": "n11"},
        ),
        state={"node_ids": torch.tensor([1, 2]), "edge_ids": torch.tensor([1])},
        metadata={"sample_id": "n11"},
        graph=graph,
    )

    sample = materialize_context(context)

    assert sample.metadata["seed"] == 11
    assert torch.equal(sample.graph.n_id, torch.tensor([11, 12]))
    assert torch.equal(sample.graph.edata["e_id"], torch.tensor([21]))
    assert sample.subgraph_seed == 0


def test_materialize_multi_seed_hetero_node_context_avoids_tensor_item(monkeypatch):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4), "y": torch.tensor([0, 1, 0])},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            written_by: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1, 2]),
            node_type="paper",
            metadata={"seed": [1, 2], "node_type": "paper", "sample_id": "p12"},
        ),
        state={
            "node_ids_by_type": {
                "paper": torch.tensor([1, 2]),
                "author": torch.tensor([0, 1]),
            }
        },
        metadata={"sample_id": "p12"},
        graph=graph,
    )

    def fail_item(self):
        raise AssertionError("hetero node materialization should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    samples = materialize_context(context)

    assert [sample.metadata["seed"] for sample in samples] == [1, 2]
    assert [sample.metadata["node_type"] for sample in samples] == ["paper", "paper"]
    assert [sample.subgraph_seed for sample in samples] == [0, 1]


def test_materialize_multi_seed_hetero_node_context_avoids_tensor_int(monkeypatch):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4), "y": torch.tensor([0, 1, 0])},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            written_by: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1, 2]),
            node_type="paper",
            metadata={"seed": [1, 2], "node_type": "paper", "sample_id": "p12"},
        ),
        state={
            "node_ids_by_type": {
                "paper": torch.tensor([1, 2]),
                "author": torch.tensor([0, 1]),
            }
        },
        metadata={"sample_id": "p12"},
        graph=graph,
    )

    def fail_int(self):
        raise AssertionError("hetero node materialization should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    samples = materialize_context(context)

    assert [sample.metadata["seed"] for sample in samples] == [1, 2]
    assert [sample.metadata["node_type"] for sample in samples] == ["paper", "paper"]
    assert [sample.subgraph_seed for sample in samples] == [0, 1]


def test_link_message_passing_graph_avoids_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.randn(3, 4),
    )

    def fail_int(self):
        raise AssertionError("link message passing graph filtering should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    message_passing_graph = _link_message_passing_graph(
        graph,
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(0),
                dst_index=torch.tensor(1),
                label=torch.tensor(1),
                metadata={"exclude_seed_edges": True},
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(2),
                dst_index=torch.tensor(0),
                label=torch.tensor(0),
            ),
        ],
    )

    assert torch.equal(message_passing_graph.edge_index, torch.tensor([[1, 1], [2, 0]]))


def test_materialize_node_context_homo_avoids_dense_node_masks(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
        y=torch.tensor([0, 1, 0, 1]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1]),
            node_type="node",
            metadata={"seed": 1, "sample_id": "n1"},
        ),
        state={"node_ids": torch.tensor([0, 1, 2, 3])},
        metadata={"sample_id": "n1"},
        graph=graph,
    )
    original_zeros = torch.zeros

    def guarded_zeros(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if kwargs.get("dtype") is torch.bool and size == graph.x.size(0) and caller is not None and caller.f_code.co_name == "_subgraph":
            raise AssertionError("homo node materialization should avoid dense bool node masks")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)

    sample = materialize_context(context)

    assert torch.equal(sample.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert sample.subgraph_seed == 1


def test_materialize_node_context_homo_avoids_dense_node_mappings(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
        y=torch.tensor([0, 1, 0, 1]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1]),
            node_type="node",
            metadata={"seed": 1, "sample_id": "n1"},
        ),
        state={"node_ids": torch.tensor([0, 1, 2, 3])},
        metadata={"sample_id": "n1"},
        graph=graph,
    )
    original_full = torch.full

    def guarded_full(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if size == (graph.x.size(0),) and caller is not None and caller.f_code.co_name == "_subgraph":
            raise AssertionError("homo node materialization should avoid dense node mappings")
        return original_full(*args, **kwargs)

    monkeypatch.setattr(torch, "full", guarded_full)

    sample = materialize_context(context)

    assert torch.equal(sample.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert sample.subgraph_seed == 1


def test_materialize_context_homo_graph_without_x_uses_fetched_node_features():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        y=torch.tensor([0, 1, 0]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1]),
            node_type="node",
            metadata={"seed": 1, "sample_id": "n1"},
        ),
        state={
            "node_ids": torch.tensor([0, 1, 2]),
            "_materialized_node_features": {
                "node": {
                    "x": SimpleNamespace(
                        index=torch.tensor([0, 1, 2]),
                        values=torch.tensor([[1.0], [2.0], [3.0]]),
                    )
                }
            },
        },
        metadata={"sample_id": "n1"},
        graph=graph,
    )

    sample = materialize_context(context)

    assert torch.equal(sample.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(sample.graph.x, torch.tensor([[1.0], [2.0], [3.0]]))
    assert sample.subgraph_seed == 1


def test_materialize_context_hetero_graph_without_x_uses_fetched_node_features():
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {"y": torch.tensor([0, 1, 0])},
            "author": {"role": torch.tensor([1, 2])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            written_by: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1]),
            node_type="paper",
            metadata={"seed": 1, "node_type": "paper", "sample_id": "p1"},
        ),
        state={
            "node_ids_by_type": {
                "paper": torch.tensor([1, 2]),
                "author": torch.tensor([0, 1]),
            },
            "_materialized_node_features": {
                "paper": {
                    "x": SimpleNamespace(
                        index=torch.tensor([1, 2]),
                        values=torch.tensor([[1.0], [2.0]]),
                    )
                },
                "author": {
                    "x": SimpleNamespace(
                        index=torch.tensor([0, 1]),
                        values=torch.tensor([[10.0], [20.0]]),
                    )
                },
            },
        },
        metadata={"sample_id": "p1"},
        graph=graph,
    )

    sample = materialize_context(context)

    assert torch.equal(sample.graph.nodes["paper"].n_id, torch.tensor([1, 2]))
    assert torch.equal(sample.graph.nodes["paper"].x, torch.tensor([[1.0], [2.0]]))
    assert torch.equal(sample.graph.nodes["author"].n_id, torch.tensor([0, 1]))
    assert torch.equal(sample.graph.nodes["author"].x, torch.tensor([[10.0], [20.0]]))
    assert sample.subgraph_seed == 0


def test_materialize_context_storage_backed_hetero_temporal_graph_without_resident_node_tensors_uses_fetched_features():
    writes = ("author", "writes", "paper")
    schema = GraphSchema(
        node_types=("author", "paper"),
        edge_types=(writes,),
        node_features={"author": (), "paper": ()},
        edge_features={writes: ("timestamp",)},
        time_attr="timestamp",
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore(
            {
                ("edge", writes, "timestamp"): InMemoryTensorStore(torch.tensor([1, 3], dtype=torch.long)),
            }
        ),
        graph_store=InMemoryGraphStore(
            {writes: torch.tensor([[0, 1], [0, 0]], dtype=torch.long)},
            num_nodes={"author": 2, "paper": 1},
        ),
    )
    context = MaterializationContext(
        request=TemporalSeedRequest(
            src_ids=torch.tensor([0]),
            dst_ids=torch.tensor([0]),
            timestamps=torch.tensor([2]),
            edge_type=writes,
        ),
        state={
            "record": TemporalEventRecord(
                graph=graph,
                src_index=0,
                dst_index=0,
                timestamp=2,
                label=1,
                edge_type=writes,
            ),
            "_materialized_node_features": {
                "author": {
                    "x": SimpleNamespace(
                        index=torch.tensor([0, 1]),
                        values=torch.tensor([[10.0], [20.0]]),
                    )
                },
                "paper": {
                    "x": SimpleNamespace(
                        index=torch.tensor([0]),
                        values=torch.tensor([[1.0]]),
                    )
                },
            },
        },
        graph=graph,
    )

    sample = materialize_context(context)

    assert torch.equal(sample.graph.nodes["author"].data["n_id"], torch.tensor([0, 1]))
    assert torch.equal(sample.graph.nodes["author"].data["x"], torch.tensor([[10.0], [20.0]]))
    assert torch.equal(sample.graph.nodes["paper"].data["n_id"], torch.tensor([0]))
    assert torch.equal(sample.graph.nodes["paper"].data["x"], torch.tensor([[1.0]]))


def test_materialize_node_context_hetero_avoids_dense_node_masks(monkeypatch):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4), "y": torch.tensor([0, 1, 0])},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            written_by: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1]),
            node_type="paper",
            metadata={"seed": 1, "node_type": "paper", "sample_id": "p1"},
        ),
        state={
            "node_ids_by_type": {
                "paper": torch.tensor([1]),
                "author": torch.tensor([0]),
            }
        },
        metadata={"sample_id": "p1"},
        graph=graph,
    )
    original_zeros = torch.zeros

    def guarded_zeros(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if kwargs.get("dtype") is torch.bool and size in {2, 3} and caller is not None and caller.f_code.co_name == "_hetero_subgraph":
            raise AssertionError("hetero node materialization should avoid dense bool node masks")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)

    sample = materialize_context(context)

    assert torch.equal(sample.graph.nodes["paper"].n_id, torch.tensor([1]))
    assert torch.equal(sample.graph.nodes["author"].n_id, torch.tensor([0]))
    assert sample.subgraph_seed == 0


def test_materialize_node_context_hetero_avoids_dense_node_mappings(monkeypatch):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4), "y": torch.tensor([0, 1, 0])},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            written_by: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([1]),
            node_type="paper",
            metadata={"seed": 1, "node_type": "paper", "sample_id": "p1"},
        ),
        state={
            "node_ids_by_type": {
                "paper": torch.tensor([1]),
                "author": torch.tensor([0]),
            }
        },
        metadata={"sample_id": "p1"},
        graph=graph,
    )
    original_full = torch.full

    def guarded_full(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if size in {(2,), (3,)} and caller is not None and caller.f_code.co_name == "_hetero_subgraph":
            raise AssertionError("hetero node materialization should avoid dense node mappings")
        return original_full(*args, **kwargs)

    monkeypatch.setattr(torch, "full", guarded_full)

    sample = materialize_context(context)

    assert torch.equal(sample.graph.nodes["paper"].n_id, torch.tensor([1]))
    assert torch.equal(sample.graph.nodes["author"].n_id, torch.tensor([0]))
    assert sample.subgraph_seed == 0


def test_build_stitched_node_samples_avoids_tensor_item(monkeypatch):
    stitched_graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.arange(6, dtype=torch.float32).view(3, 2),
        n_id=torch.tensor([10, 11, 12]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([11]),
            node_type="node",
            metadata={"seed": 11, "sample_id": "n11"},
        ),
        metadata={"sample_id": "n11"},
    )

    def fail_item(self):
        raise AssertionError("stitched homo node sample building should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    sample = _build_stitched_node_samples(
        context,
        stitched_graph,
        torch.tensor([1]),
        torch.tensor([11]),
    )

    assert sample.metadata["seed"] == 11
    assert sample.subgraph_seed == 1


def test_build_stitched_node_samples_avoids_tensor_int(monkeypatch):
    stitched_graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.arange(6, dtype=torch.float32).view(3, 2),
        n_id=torch.tensor([10, 11, 12]),
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([11]),
            node_type="node",
            metadata={"seed": 11, "sample_id": "n11"},
        ),
        metadata={"sample_id": "n11"},
    )

    def fail_int(self):
        raise AssertionError("stitched homo node sample building should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    sample = _build_stitched_node_samples(
        context,
        stitched_graph,
        torch.tensor([1]),
        torch.tensor([11]),
    )

    assert sample.metadata["seed"] == 11
    assert sample.subgraph_seed == 1


def test_build_stitched_hetero_node_samples_avoids_tensor_item(monkeypatch):
    writes = ("author", "writes", "paper")
    stitched_graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0]]), "n_id": torch.tensor([20])},
            "paper": {"x": torch.tensor([[1.0], [2.0]]), "n_id": torch.tensor([30, 31])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0], [1]])},
        },
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([31]),
            node_type="paper",
            metadata={"seed": 31, "node_type": "paper", "sample_id": "p31"},
        ),
        metadata={"sample_id": "p31"},
    )

    def fail_item(self):
        raise AssertionError("stitched hetero node sample building should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    sample = _build_stitched_hetero_node_samples(
        context,
        stitched_graph,
        torch.tensor([1]),
        torch.tensor([31]),
        node_type="paper",
    )

    assert sample.metadata["seed"] == 31
    assert sample.metadata["node_type"] == "paper"
    assert sample.subgraph_seed == 1


def test_build_stitched_hetero_node_samples_avoids_tensor_int(monkeypatch):
    writes = ("author", "writes", "paper")
    stitched_graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0]]), "n_id": torch.tensor([20])},
            "paper": {"x": torch.tensor([[1.0], [2.0]]), "n_id": torch.tensor([30, 31])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0], [1]])},
        },
    )
    context = MaterializationContext(
        request=NodeSeedRequest(
            node_ids=torch.tensor([31]),
            node_type="paper",
            metadata={"seed": 31, "node_type": "paper", "sample_id": "p31"},
        ),
        metadata={"sample_id": "p31"},
    )

    def fail_int(self):
        raise AssertionError("stitched hetero node sample building should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    sample = _build_stitched_hetero_node_samples(
        context,
        stitched_graph,
        torch.tensor([1]),
        torch.tensor([31]),
        node_type="paper",
    )

    assert sample.metadata["seed"] == 31
    assert sample.metadata["node_type"] == "paper"
    assert sample.subgraph_seed == 1
