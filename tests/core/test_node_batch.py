import torch

from vgl import Graph
from vgl.core.batch import NodeBatch
from vgl.data.sample import SampleRecord

WRITES = ("author", "writes", "paper")
CITES = ("paper", "cites", "paper")


def _sample(graph, seed, sample_id):
    return SampleRecord(
        graph=graph,
        metadata={"seed": seed, "sample_id": sample_id},
        sample_id=sample_id,
        subgraph_seed=seed,
    )


def test_node_batch_batches_subgraphs_into_disjoint_union():
    g1 = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    g2 = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
        y=torch.tensor([1, 0, 1]),
    )

    batch = NodeBatch.from_samples([_sample(g1, 1, "a"), _sample(g2, 2, "b")])

    assert batch.graph.x.size(0) == 5
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 2, 3], [1, 3, 4]]))
    assert torch.equal(batch.seed_index, torch.tensor([1, 4]))
    assert batch.metadata == [{"seed": 1, "sample_id": "a"}, {"seed": 2, "sample_id": "b"}]


def test_node_batch_metadata_preserves_sample_record_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
    )

    batch = NodeBatch.from_samples(
        [SampleRecord(graph=graph, metadata={}, sample_id="a", source_graph_id="root-a", subgraph_seed=0)]
    )

    assert batch.metadata == [{"sample_id": "a", "source_graph_id": "root-a"}]


def test_node_batch_batches_subgraphs_without_x():
    g1 = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        y=torch.tensor([0, 1]),
    )
    g2 = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([1, 0, 1]),
    )

    batch = NodeBatch.from_samples([_sample(g1, 1, "a"), _sample(g2, 2, "b")])

    assert torch.equal(batch.seed_index, torch.tensor([1, 4]))
    assert batch.metadata == [{"seed": 1, "sample_id": "a"}, {"seed": 2, "sample_id": "b"}]


def test_node_batch_batches_hetero_subgraphs_with_seed_type_offsets():
    g1 = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(2, 4), "y": torch.tensor([0, 1])},
            "author": {"x": torch.randn(1, 4)},
        },
        edges={
            ("author", "writes", "paper"): {"edge_index": torch.tensor([[0], [1]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1], [0]])},
        },
    )
    g2 = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(1, 4), "y": torch.tensor([1])},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {"edge_index": torch.tensor([[1], [0]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[0], [1]])},
        },
    )

    batch = NodeBatch.from_samples(
        [
            SampleRecord(graph=g1, metadata={"seed": 1, "node_type": "paper"}, sample_id="a", subgraph_seed=1),
            SampleRecord(graph=g2, metadata={"seed": 0, "node_type": "paper"}, sample_id="b", subgraph_seed=0),
        ]
    )

    assert batch.graph.nodes["paper"].x.size(0) == 3
    assert batch.graph.nodes["author"].x.size(0) == 3
    assert torch.equal(batch.seed_index, torch.tensor([1, 2]))
    assert torch.equal(
        batch.graph.edges[("author", "writes", "paper")].edge_index,
        torch.tensor([[0, 2], [1, 2]]),
    )


def test_node_batch_batches_block_layers_across_samples():
    g1 = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        x=torch.randn(3, 4),
        n_id=torch.tensor([10, 11, 12], dtype=torch.long),
        edge_data={"e_id": torch.tensor([100, 101], dtype=torch.long)},
    )
    g2 = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        x=torch.randn(3, 4),
        n_id=torch.tensor([20, 21, 22], dtype=torch.long),
        edge_data={"e_id": torch.tensor([200, 201], dtype=torch.long)},
    )
    g1_blocks = [g1.to_block(torch.tensor([1, 2], dtype=torch.long)), g1.to_block(torch.tensor([2], dtype=torch.long))]
    g2_blocks = [g2.to_block(torch.tensor([1, 2], dtype=torch.long)), g2.to_block(torch.tensor([2], dtype=torch.long))]

    batch = NodeBatch.from_samples(
        [
            SampleRecord(
                graph=g1,
                metadata={"seed": 2, "sample_id": "a"},
                sample_id="a",
                subgraph_seed=2,
                blocks=g1_blocks,
            ),
            SampleRecord(
                graph=g2,
                metadata={"seed": 2, "sample_id": "b"},
                sample_id="b",
                subgraph_seed=2,
                blocks=g2_blocks,
            ),
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
    outer_offset = torch.tensor(
        [[g1_blocks[0].src_n_id.numel()], [g1_blocks[0].dst_n_id.numel()]],
        dtype=torch.long,
    )
    inner_offset = torch.tensor(
        [[g1_blocks[1].src_n_id.numel()], [g1_blocks[1].dst_n_id.numel()]],
        dtype=torch.long,
    )
    assert torch.equal(
        outer_block.edge_index,
        torch.cat([g1_blocks[0].edge_index, g2_blocks[0].edge_index + outer_offset], dim=1),
    )
    assert torch.equal(
        inner_block.edge_index,
        torch.cat([g1_blocks[1].edge_index, g2_blocks[1].edge_index + inner_offset], dim=1),
    )


def test_node_batch_batches_hetero_block_layers_across_samples():
    g1 = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4), "n_id": torch.tensor([10, 12], dtype=torch.long)},
            "paper": {"x": torch.randn(2, 4), "n_id": torch.tensor([20, 21], dtype=torch.long)},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.long),
                "e_id": torch.tensor([100, 101, 102], dtype=torch.long),
            }
        },
    )
    g2 = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4), "n_id": torch.tensor([30, 32], dtype=torch.long)},
            "paper": {"x": torch.randn(2, 4), "n_id": torch.tensor([40, 41], dtype=torch.long)},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.long),
                "e_id": torch.tensor([200, 201, 202], dtype=torch.long),
            }
        },
    )
    g1_blocks = [
        g1.to_block(torch.tensor([0, 1], dtype=torch.long), edge_type=WRITES),
        g1.to_block(torch.tensor([0], dtype=torch.long), edge_type=WRITES),
    ]
    g2_blocks = [
        g2.to_block(torch.tensor([0, 1], dtype=torch.long), edge_type=WRITES),
        g2.to_block(torch.tensor([0], dtype=torch.long), edge_type=WRITES),
    ]

    batch = NodeBatch.from_samples(
        [
            SampleRecord(
                graph=g1,
                metadata={"seed": 0, "sample_id": "a", "node_type": "paper"},
                sample_id="a",
                subgraph_seed=0,
                blocks=g1_blocks,
            ),
            SampleRecord(
                graph=g2,
                metadata={"seed": 0, "sample_id": "b", "node_type": "paper"},
                sample_id="b",
                subgraph_seed=0,
                blocks=g2_blocks,
            ),
        ]
    )

    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block = batch.blocks[0]
    inner_block = batch.blocks[1]
    assert outer_block.edge_type == WRITES
    assert torch.equal(outer_block.src_n_id, torch.cat([g1_blocks[0].src_n_id, g2_blocks[0].src_n_id], dim=0))
    assert torch.equal(outer_block.dst_n_id, torch.cat([g1_blocks[0].dst_n_id, g2_blocks[0].dst_n_id], dim=0))
    assert torch.equal(inner_block.src_n_id, torch.cat([g1_blocks[1].src_n_id, g2_blocks[1].src_n_id], dim=0))
    assert torch.equal(inner_block.dst_n_id, torch.cat([g1_blocks[1].dst_n_id, g2_blocks[1].dst_n_id], dim=0))
    outer_offset = torch.tensor(
        [[g1_blocks[0].src_n_id.numel()], [g1_blocks[0].dst_n_id.numel()]],
        dtype=torch.long,
    )
    inner_offset = torch.tensor(
        [[g1_blocks[1].src_n_id.numel()], [g1_blocks[1].dst_n_id.numel()]],
        dtype=torch.long,
    )
    assert torch.equal(
        outer_block.edge_index,
        torch.cat([g1_blocks[0].edge_index, g2_blocks[0].edge_index + outer_offset], dim=1),
    )
    assert torch.equal(
        inner_block.edge_index,
        torch.cat([g1_blocks[1].edge_index, g2_blocks[1].edge_index + inner_offset], dim=1),
    )


def test_node_batch_batches_multi_relation_hetero_block_layers_across_samples():
    g1 = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4), "n_id": torch.tensor([10, 12], dtype=torch.long)},
            "paper": {"x": torch.randn(3, 4), "n_id": torch.tensor([20, 21, 22], dtype=torch.long)},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long),
                "e_id": torch.tensor([100, 101, 102], dtype=torch.long),
            },
            CITES: {
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
            WRITES: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long),
                "e_id": torch.tensor([200, 201, 202], dtype=torch.long),
            },
            CITES: {
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

    batch = NodeBatch.from_samples(
        [
            SampleRecord(
                graph=g1,
                metadata={"seed": 0, "sample_id": "a", "node_type": "paper"},
                sample_id="a",
                subgraph_seed=0,
                blocks=g1_blocks,
            ),
            SampleRecord(
                graph=g2,
                metadata={"seed": 0, "sample_id": "b", "node_type": "paper"},
                sample_id="b",
                subgraph_seed=0,
                blocks=g2_blocks,
            ),
        ]
    )

    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block = batch.blocks[0]
    inner_block = batch.blocks[1]
    assert outer_block.edge_types == (WRITES, CITES)
    assert torch.equal(
        outer_block.src_n_id["author"],
        torch.cat([g1_blocks[0].src_n_id["author"], g2_blocks[0].src_n_id["author"]], dim=0),
    )
    assert torch.equal(
        outer_block.src_n_id["paper"],
        torch.cat([g1_blocks[0].src_n_id["paper"], g2_blocks[0].src_n_id["paper"]], dim=0),
    )
    assert torch.equal(
        outer_block.dst_n_id["paper"],
        torch.cat([g1_blocks[0].dst_n_id["paper"], g2_blocks[0].dst_n_id["paper"]], dim=0),
    )
    writes_offset = torch.tensor(
        [
            [g1_blocks[0].src_n_id["author"].numel()],
            [g1_blocks[0].dst_n_id["paper"].numel()],
        ],
        dtype=torch.long,
    )
    cites_offset = torch.tensor(
        [
            [g1_blocks[0].src_n_id["paper"].numel()],
            [g1_blocks[0].dst_n_id["paper"].numel()],
        ],
        dtype=torch.long,
    )
    assert torch.equal(
        outer_block.edge_index(WRITES),
        torch.cat([g1_blocks[0].edge_index(WRITES), g2_blocks[0].edge_index(WRITES) + writes_offset], dim=1),
    )
    assert torch.equal(
        outer_block.edge_index(CITES),
        torch.cat([g1_blocks[0].edge_index(CITES), g2_blocks[0].edge_index(CITES) + cites_offset], dim=1),
    )
    assert torch.equal(
        inner_block.dst_n_id["paper"],
        torch.cat([g1_blocks[1].dst_n_id["paper"], g2_blocks[1].dst_n_id["paper"]], dim=0),
    )
