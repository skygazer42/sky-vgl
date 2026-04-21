import pytest
import torch

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch
from vgl.data.sample import LinkPredictionRecord
from vgl.ops import to_block


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )


def test_link_prediction_batch_tracks_fields():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, sample_id="p", query_id="q0"),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, sample_id="n", query_id="q1"),
        ]
    )

    assert batch.graph is graph
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))
    assert batch.metadata == [
        {"sample_id": "p", "query_id": "q0", "edge_type": ("node", "to", "node")},
        {"sample_id": "n", "query_id": "q1", "edge_type": ("node", "to", "node")},
    ]


def test_link_prediction_batch_rejects_empty_records():
    with pytest.raises(ValueError, match="at least one record"):
        LinkPredictionBatch.from_records([])


def test_link_prediction_batch_batches_mixed_graphs_into_a_disjoint_union():
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=_graph(), src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=_graph(), src_index=1, dst_index=2, label=0),
        ]
    )

    assert torch.equal(batch.src_index, torch.tensor([0, 4]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 5]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))
    assert batch.graph.x.size(0) == 6
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]))


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


def test_link_prediction_batch_can_exclude_seed_edges_from_message_passing_graph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.randn(3, 4),
    )
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                metadata={"exclude_seed_edges": True},
            ),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
        ]
    )

    assert batch.graph is not graph
    assert torch.equal(batch.graph.edge_index, torch.tensor([[1, 1], [2, 0]]))
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))


def test_link_prediction_batch_excludes_seed_edges_without_tensor_int_conversion(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.randn(3, 4),
    )

    def fail_int(self):
        raise AssertionError("link prediction batching should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    batch = LinkPredictionBatch.from_records(
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
        ]
    )

    assert batch.graph is not graph
    assert torch.equal(batch.graph.edge_index, torch.tensor([[1, 1], [2, 0]]))
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))


def test_link_prediction_batch_tracks_filter_mask():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, query_id="q0"),
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=2,
                label=0,
                query_id="q0",
                filter_ranking=True,
            ),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, query_id="q0"),
        ]
    )

    assert torch.equal(batch.query_index, torch.tensor([0, 0, 0]))
    assert torch.equal(batch.filter_mask, torch.tensor([False, True, False]))


def test_link_prediction_batch_builds_query_index_from_metadata_backed_query_ids():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, metadata={"query_id": "q0"}),
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=2, label=0, metadata={"query_id": "q0"}),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, metadata={"query_id": "q1"}),
        ]
    )

    assert torch.equal(batch.query_index, torch.tensor([0, 0, 1]))


def test_link_prediction_batch_falls_back_to_sample_ids_for_query_index():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, sample_id="q0"),
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=2, label=0, sample_id="q0"),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, sample_id="q1"),
        ]
    )

    assert torch.equal(batch.query_index, torch.tensor([0, 0, 1]))


def test_link_prediction_batch_ignores_none_metadata_query_ids_and_still_falls_back_to_sample_id():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, sample_id="q0", metadata={"query_id": None}),
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=2, label=0, sample_id="q0", metadata={"query_id": None}),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, sample_id="q1", metadata={"query_id": None}),
        ]
    )

    assert torch.equal(batch.query_index, torch.tensor([0, 0, 1]))


def test_link_prediction_batch_can_fall_back_to_metadata_backed_sample_ids_for_query_index():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, metadata={"sample_id": "q0"}),
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=2, label=0, metadata={"sample_id": "q0"}),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, metadata={"sample_id": "q1"}),
        ]
    )

    assert torch.equal(batch.query_index, torch.tensor([0, 0, 1]))


def test_link_prediction_batch_batches_hetero_graphs_for_single_edge_type():
    edge_type = ("author", "writes", "paper")
    g1 = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    g2 = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(1, 4)},
            "paper": {"x": torch.randn(2, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0], [1]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1], [0]])},
        },
    )

    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=g1, src_index=0, dst_index=1, label=1, edge_type=edge_type),
            LinkPredictionRecord(graph=g2, src_index=0, dst_index=1, label=0, edge_type=edge_type),
        ]
    )

    assert batch.edge_type == edge_type
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 4]))
    assert torch.equal(
        batch.graph.edges[edge_type].edge_index,
        torch.tensor([[0, 1, 2], [1, 2, 4]]),
    )


def test_link_prediction_batch_can_exclude_seed_and_reverse_edges_for_hetero_records():
    edge_type = ("author", "writes", "paper")
    reverse_edge_type = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            reverse_edge_type: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                edge_type=edge_type,
                reverse_edge_type=reverse_edge_type,
                metadata={"exclude_seed_edges": True},
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=1,
                dst_index=2,
                label=0,
                edge_type=edge_type,
                reverse_edge_type=reverse_edge_type,
            ),
        ]
    )

    assert torch.equal(
        batch.graph.edges[edge_type].edge_index,
        torch.tensor([[1], [2]]),
    )
    assert torch.equal(
        batch.graph.edges[reverse_edge_type].edge_index,
        torch.tensor([[2], [1]]),
    )


def test_link_prediction_batch_supports_mixed_hetero_edge_types():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 2], [0, 1]])},
            cites: {"edge_index": torch.tensor([[0, 2], [2, 3]])},
        },
    )

    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, edge_type=writes),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=3, label=0, edge_type=cites),
        ]
    )

    assert batch.edge_type is None
    assert batch.edge_types == (writes, cites)
    assert torch.equal(batch.edge_type_index, torch.tensor([0, 1]))
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 3]))


def test_link_prediction_batch_rejects_block_batches_with_mixed_hetero_edge_types():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 2], [0, 1]])},
            cites: {"edge_index": torch.tensor([[0, 2], [2, 3]])},
        },
    )
    writes_block = to_block(graph, torch.tensor([1]), edge_type=writes)
    cites_block = to_block(graph, torch.tensor([2, 3]), edge_type=cites)

    with pytest.raises(ValueError, match="single edge_type"):
        LinkPredictionBatch.from_records(
            [
                LinkPredictionRecord(
                    graph=graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    edge_type=writes,
                    blocks=[writes_block],
                ),
                LinkPredictionRecord(
                    graph=graph,
                    src_index=2,
                    dst_index=3,
                    label=0,
                    edge_type=cites,
                    blocks=[cites_block],
                ),
            ]
        )


def test_link_prediction_batch_excludes_seed_edges_for_each_relation_in_mixed_hetero_batch():
    writes = ("author", "writes", "paper")
    reverse_writes = ("paper", "written_by", "author")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            reverse_writes: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
            cites: {"edge_index": torch.tensor([[0, 2], [2, 3]])},
        },
    )
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                edge_type=writes,
                reverse_edge_type=reverse_writes,
                metadata={"exclude_seed_edges": True},
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=2,
                dst_index=3,
                label=1,
                edge_type=cites,
                metadata={"exclude_seed_edges": True},
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=1,
                dst_index=2,
                label=0,
                edge_type=writes,
            ),
        ]
    )

    assert torch.equal(batch.graph.edges[writes].edge_index, torch.tensor([[1], [2]]))
    assert torch.equal(batch.graph.edges[reverse_writes].edge_index, torch.tensor([[2], [1]]))
    assert torch.equal(batch.graph.edges[cites].edge_index, torch.tensor([[0], [2]]))


def test_link_prediction_batch_batches_block_layers_across_records():
    g1 = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        x=torch.randn(4, 4),
        n_id=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        edge_data={"e_id": torch.tensor([100, 101, 102], dtype=torch.long)},
    )
    g2 = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        x=torch.randn(4, 4),
        n_id=torch.tensor([20, 21, 22, 23], dtype=torch.long),
        edge_data={"e_id": torch.tensor([200, 201, 202], dtype=torch.long)},
    )
    g1_blocks = [g1.to_block(torch.tensor([1, 2, 3], dtype=torch.long)), g1.to_block(torch.tensor([2, 3], dtype=torch.long))]
    g2_blocks = [g2.to_block(torch.tensor([1, 2, 3], dtype=torch.long)), g2.to_block(torch.tensor([2, 3], dtype=torch.long))]

    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=g1, src_index=1, dst_index=2, label=1, blocks=g1_blocks),
            LinkPredictionRecord(graph=g2, src_index=1, dst_index=2, label=0, blocks=g2_blocks),
        ]
    )

    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block = batch.blocks[0]
    inner_block = batch.blocks[1]
    assert torch.equal(outer_block.src_n_id, torch.cat([g1_blocks[0].src_n_id, g2_blocks[0].src_n_id], dim=0))
    assert torch.equal(outer_block.dst_n_id, torch.cat([g1_blocks[0].dst_n_id, g2_blocks[0].dst_n_id], dim=0))
    assert torch.equal(inner_block.src_n_id, torch.cat([g1_blocks[1].src_n_id, g2_blocks[1].src_n_id], dim=0))
    assert torch.equal(inner_block.dst_n_id, torch.cat([g1_blocks[1].dst_n_id, g2_blocks[1].dst_n_id], dim=0))


def test_link_prediction_batch_batches_multi_relation_hetero_block_layers_across_records():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    g1 = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4), "n_id": torch.tensor([10, 12], dtype=torch.long)},
            "paper": {"x": torch.randn(3, 4), "n_id": torch.tensor([20, 21, 22], dtype=torch.long)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long),
                "e_id": torch.tensor([100, 101, 102], dtype=torch.long),
            },
            cites: {
                "edge_index": torch.tensor([[0, 1, 2], [2, 2, 0]], dtype=torch.long),
                "e_id": torch.tensor([110, 111, 112], dtype=torch.long),
            },
        },
    )
    g2 = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4), "n_id": torch.tensor([30, 32], dtype=torch.long)},
            "paper": {"x": torch.randn(3, 4), "n_id": torch.tensor([40, 41, 42], dtype=torch.long)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long),
                "e_id": torch.tensor([200, 201, 202], dtype=torch.long),
            },
            cites: {
                "edge_index": torch.tensor([[0, 1, 2], [2, 2, 0]], dtype=torch.long),
                "e_id": torch.tensor([210, 211, 212], dtype=torch.long),
            },
        },
    )
    g1_blocks = [
        g1.to_hetero_block({"paper": torch.tensor([0, 2], dtype=torch.long)}),
        g1.to_hetero_block({"paper": torch.tensor([0], dtype=torch.long)}),
    ]
    g2_blocks = [
        g2.to_hetero_block({"paper": torch.tensor([0, 2], dtype=torch.long)}),
        g2.to_hetero_block({"paper": torch.tensor([0], dtype=torch.long)}),
    ]

    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=g1, src_index=1, dst_index=0, label=1, edge_type=writes, blocks=g1_blocks),
            LinkPredictionRecord(graph=g2, src_index=1, dst_index=0, label=0, edge_type=writes, blocks=g2_blocks),
        ]
    )

    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block = batch.blocks[0]
    inner_block = batch.blocks[1]
    assert outer_block.edge_types == (writes, cites)
    assert torch.equal(
        outer_block.src_n_id["author"],
        torch.cat([g1_blocks[0].src_n_id["author"], g2_blocks[0].src_n_id["author"]], dim=0),
    )
    assert torch.equal(
        outer_block.src_n_id["paper"],
        torch.cat([g1_blocks[0].src_n_id["paper"], g2_blocks[0].src_n_id["paper"]], dim=0),
    )
    assert torch.equal(
        inner_block.dst_n_id["paper"],
        torch.cat([g1_blocks[1].dst_n_id["paper"], g2_blocks[1].dst_n_id["paper"]], dim=0),
    )
