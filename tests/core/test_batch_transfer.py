import torch

from vgl import Graph
from vgl.core.batch import GraphBatch, LinkPredictionBatch, NodeBatch, TemporalEventBatch


def _homo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        x=torch.randn(2, 4),
    )


def _temporal_graph():
    return Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                "timestamp": torch.tensor([1.0, 2.0]),
            }
        },
        time_attr="timestamp",
    )


def _transfer_device():
    return torch.device("meta")


def test_graph_batch_to_moves_graphs_and_batch_tensors():
    graph_one = _homo_graph()
    graph_two = _homo_graph()
    metadata = [{"sample_id": "g0"}, {"sample_id": "g1"}]
    batch = GraphBatch(
        graphs=[graph_one, graph_two],
        graph_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        graph_ptr=torch.tensor([0, 2, 4], dtype=torch.long),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        metadata=metadata,
    )

    moved = batch.to(device=_transfer_device(), dtype=torch.float64)

    assert moved is not batch
    assert moved.graphs[0] is not graph_one
    assert moved.graphs[1] is not graph_two
    assert moved.graphs[0].x.device.type == "meta"
    assert moved.graphs[1].x.device.type == "meta"
    assert moved.graph_index.device.type == "meta"
    assert moved.graph_ptr is not None
    assert moved.graph_ptr.device.type == "meta"
    assert moved.graph_ptr.dtype == torch.long
    assert moved.labels is not None
    assert moved.labels.device.type == "meta"
    assert moved.labels.dtype == torch.float64
    assert moved.metadata is metadata
    assert graph_one.x.dtype == torch.float32
    assert graph_two.x.dtype == torch.float32


def test_graph_batch_pin_memory_pins_graphs_and_batch_tensors():
    batch = GraphBatch(
        graphs=[_homo_graph(), _homo_graph()],
        graph_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        graph_ptr=torch.tensor([0, 2, 4], dtype=torch.long),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        metadata=[{"sample_id": "g0"}, {"sample_id": "g1"}],
    )

    pinned = batch.pin_memory()

    assert pinned is not batch
    assert pinned.graphs[0].x.is_pinned()
    assert pinned.graphs[1].x.is_pinned()
    assert pinned.graph_index.is_pinned()
    assert pinned.graph_ptr is not None
    assert pinned.graph_ptr.is_pinned()
    assert pinned.labels is not None
    assert pinned.labels.is_pinned()
    assert pinned.metadata is batch.metadata
    assert not batch.graph_index.is_pinned()


def _hetero_graph():
    return Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(2, 4)},
            "author": {"x": torch.randn(1, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
            },
        },
    )


def test_graph_batch_to_moves_typed_membership_tensors_for_hetero_batches():
    graph_one = _hetero_graph()
    graph_two = _hetero_graph()
    metadata = [{"sample_id": "h0"}, {"sample_id": "h1"}]
    batch = GraphBatch(
        graphs=[graph_one, graph_two],
        graph_index=None,
        graph_ptr=None,
        graph_index_by_type={
            "paper": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "author": torch.tensor([0, 1], dtype=torch.long),
        },
        graph_ptr_by_type={
            "paper": torch.tensor([0, 2, 4], dtype=torch.long),
            "author": torch.tensor([0, 1, 2], dtype=torch.long),
        },
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        metadata=metadata,
    )

    moved = batch.to(device=_transfer_device(), dtype=torch.float64)

    assert moved is not batch
    assert moved.graphs[0] is not graph_one
    assert moved.graphs[0].nodes["paper"].x.device.type == "meta"
    assert moved.graph_index is None
    assert moved.graph_ptr is None
    assert moved.graph_index_by_type["paper"].device.type == "meta"
    assert moved.graph_index_by_type["paper"].dtype == torch.long
    assert moved.graph_ptr_by_type["author"].device.type == "meta"
    assert moved.graph_ptr_by_type["author"].dtype == torch.long
    assert moved.labels is not None
    assert moved.labels.device.type == "meta"
    assert moved.labels.dtype == torch.float64
    assert moved.metadata is metadata


def test_graph_batch_pin_memory_pins_typed_membership_tensors_for_hetero_batches():
    batch = GraphBatch(
        graphs=[_hetero_graph(), _hetero_graph()],
        graph_index=None,
        graph_ptr=None,
        graph_index_by_type={
            "paper": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "author": torch.tensor([0, 1], dtype=torch.long),
        },
        graph_ptr_by_type={
            "paper": torch.tensor([0, 2, 4], dtype=torch.long),
            "author": torch.tensor([0, 1, 2], dtype=torch.long),
        },
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        metadata=[{"sample_id": "h0"}, {"sample_id": "h1"}],
    )

    pinned = batch.pin_memory()

    assert pinned is not batch
    assert pinned.graphs[0].nodes["paper"].x.is_pinned()
    assert pinned.graph_index is None
    assert pinned.graph_ptr is None
    assert pinned.graph_index_by_type["paper"].is_pinned()
    assert pinned.graph_ptr_by_type["author"].is_pinned()
    assert pinned.labels is not None
    assert pinned.labels.is_pinned()
    assert pinned.metadata is batch.metadata


def test_node_batch_to_moves_graph_and_seed_index():
    graph = _homo_graph()
    metadata = [{"seed": 0}, {"seed": 1}]
    batch = NodeBatch(
        graph=graph,
        seed_index=torch.tensor([0, 1], dtype=torch.long),
        metadata=metadata,
    )

    moved = batch.to(device=_transfer_device(), dtype=torch.float64)

    assert moved is not batch
    assert moved.graph is not graph
    assert moved.graph.x.device.type == "meta"
    assert moved.graph.x.dtype == torch.float64
    assert moved.seed_index.device.type == "meta"
    assert moved.seed_index.dtype == torch.long
    assert moved.metadata is metadata
    assert graph.x.dtype == torch.float32


def test_node_batch_pin_memory_pins_graph_and_seed_index():
    batch = NodeBatch(
        graph=_homo_graph(),
        seed_index=torch.tensor([0, 1], dtype=torch.long),
        metadata=[{"seed": 0}, {"seed": 1}],
    )

    pinned = batch.pin_memory()

    assert pinned is not batch
    assert pinned.graph.x.is_pinned()
    assert pinned.seed_index.is_pinned()
    assert pinned.metadata is batch.metadata
    assert not batch.seed_index.is_pinned()


def test_link_prediction_batch_to_moves_all_transfer_fields():
    graph = _homo_graph()
    metadata = [{"query_id": "q0"}, {"query_id": "q1"}]
    batch = LinkPredictionBatch(
        graph=graph,
        src_index=torch.tensor([0, 1], dtype=torch.long),
        dst_index=torch.tensor([1, 0], dtype=torch.long),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        query_index=torch.tensor([0, 1], dtype=torch.long),
        filter_mask=torch.tensor([False, True], dtype=torch.bool),
        metadata=metadata,
    )

    moved = batch.to(device=_transfer_device(), dtype=torch.float64)

    assert moved is not batch
    assert moved.graph is not graph
    assert moved.graph.x.device.type == "meta"
    assert moved.graph.x.dtype == torch.float64
    assert moved.src_index.device.type == "meta"
    assert moved.src_index.dtype == torch.long
    assert moved.dst_index.device.type == "meta"
    assert moved.dst_index.dtype == torch.long
    assert moved.labels.device.type == "meta"
    assert moved.labels.dtype == torch.float64
    assert moved.query_index is not None
    assert moved.query_index.device.type == "meta"
    assert moved.query_index.dtype == torch.long
    assert moved.filter_mask is not None
    assert moved.filter_mask.device.type == "meta"
    assert moved.filter_mask.dtype == torch.bool
    assert moved.metadata is metadata
    assert graph.x.dtype == torch.float32


def test_link_prediction_batch_pin_memory_pins_all_transfer_fields():
    batch = LinkPredictionBatch(
        graph=_homo_graph(),
        src_index=torch.tensor([0, 1], dtype=torch.long),
        dst_index=torch.tensor([1, 0], dtype=torch.long),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        query_index=torch.tensor([0, 1], dtype=torch.long),
        filter_mask=torch.tensor([False, True], dtype=torch.bool),
        metadata=[{"query_id": "q0"}, {"query_id": "q1"}],
    )

    pinned = batch.pin_memory()

    assert pinned is not batch
    assert pinned.graph.x.is_pinned()
    assert pinned.src_index.is_pinned()
    assert pinned.dst_index.is_pinned()
    assert pinned.labels.is_pinned()
    assert pinned.query_index is not None
    assert pinned.query_index.is_pinned()
    assert pinned.filter_mask is not None
    assert pinned.filter_mask.is_pinned()
    assert pinned.metadata is batch.metadata
    assert not batch.src_index.is_pinned()


def test_temporal_event_batch_to_moves_all_transfer_fields():
    graph = _temporal_graph()
    metadata = [{"event_id": "e0"}, {"event_id": "e1"}]
    batch = TemporalEventBatch(
        graph=graph,
        src_index=torch.tensor([0, 1], dtype=torch.long),
        dst_index=torch.tensor([1, 2], dtype=torch.long),
        timestamp=torch.tensor([1.0, 2.0], dtype=torch.float32),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        event_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        metadata=metadata,
    )

    moved = batch.to(device=_transfer_device(), dtype=torch.float64)

    assert moved is not batch
    assert moved.graph is not graph
    assert moved.graph.nodes["node"].x.device.type == "meta"
    assert moved.graph.nodes["node"].x.dtype == torch.float64
    assert moved.src_index.device.type == "meta"
    assert moved.src_index.dtype == torch.long
    assert moved.dst_index.device.type == "meta"
    assert moved.dst_index.dtype == torch.long
    assert moved.timestamp.device.type == "meta"
    assert moved.timestamp.dtype == torch.float64
    assert moved.labels.device.type == "meta"
    assert moved.labels.dtype == torch.float64
    assert moved.event_features is not None
    assert moved.event_features.device.type == "meta"
    assert moved.event_features.dtype == torch.float64
    assert moved.metadata is metadata
    assert graph.nodes["node"].x.dtype == torch.float32


def test_temporal_event_batch_pin_memory_pins_all_transfer_fields():
    batch = TemporalEventBatch(
        graph=_temporal_graph(),
        src_index=torch.tensor([0, 1], dtype=torch.long),
        dst_index=torch.tensor([1, 2], dtype=torch.long),
        timestamp=torch.tensor([1.0, 2.0], dtype=torch.float32),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        event_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        metadata=[{"event_id": "e0"}, {"event_id": "e1"}],
    )

    pinned = batch.pin_memory()

    assert pinned is not batch
    assert pinned.graph.nodes["node"].x.is_pinned()
    assert pinned.src_index.is_pinned()
    assert pinned.dst_index.is_pinned()
    assert pinned.timestamp.is_pinned()
    assert pinned.labels.is_pinned()
    assert pinned.event_features is not None
    assert pinned.event_features.is_pinned()
    assert pinned.metadata is batch.metadata
    assert not batch.src_index.is_pinned()
