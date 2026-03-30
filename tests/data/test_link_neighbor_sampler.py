import torch

from vgl import Graph, HeteroBlock
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import LinkPredictionRecord
from vgl.data.sampler import LinkNeighborSampler, UniformNegativeLinkSampler
from vgl.distributed import LocalGraphShard, LocalSamplingCoordinator, write_partitioned_graph
from vgl.distributed.coordinator import StoreBackedSamplingCoordinator
from vgl.storage import FeatureStore, InMemoryTensorStore


HOMO_EDGE = ("node", "to", "node")
WRITES = ("author", "writes", "paper")
WRITTEN_BY = ("paper", "written_by", "author")


def _store_backed_coordinator(shards):
    root = next(iter(shards.values())).root
    return StoreBackedSamplingCoordinator.from_partition_dir(root)


def test_link_neighbor_sampler_extracts_local_subgraph_with_global_node_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])

    record = sampler.sample(LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1))

    assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(record.graph.edge_index, torch.tensor([[0, 1, 3], [1, 2, 1]]))
    assert record.src_index == 0
    assert record.dst_index == 1


def test_link_neighbor_sampler_can_wrap_uniform_negative_sampling():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.randn(5, 4),
    )
    torch.manual_seed(0)
    sampler = LinkNeighborSampler(
        num_neighbors=[-1],
        base_sampler=UniformNegativeLinkSampler(num_negatives=2),
    )

    records = sampler.sample(LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1))

    assert len(records) == 3
    assert all(record.graph is records[0].graph for record in records)
    assert torch.equal(records[0].graph.n_id, torch.tensor([0, 1, 2, 4]))
    assert [int(record.label) for record in records] == [1, 0, 0]
    assert all(0 <= int(record.src_index) < records[0].graph.x.size(0) for record in records)
    assert all(0 <= int(record.dst_index) < records[0].graph.x.size(0) for record in records)


def test_link_neighbor_sampler_extracts_hetero_local_subgraph_for_single_edge_type():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 2], [0, 1]])
            },
        },
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])

    record = sampler.sample(
        LinkPredictionRecord(
            graph=graph,
            src_index=0,
            dst_index=1,
            label=1,
            edge_type=("author", "writes", "paper"),
        )
    )

    assert record.edge_type == ("author", "writes", "paper")
    assert torch.equal(record.graph.nodes["author"].n_id, torch.tensor([0]))
    assert torch.equal(record.graph.nodes["paper"].n_id, torch.tensor([1]))
    assert torch.equal(
        record.graph.edges[("author", "writes", "paper")].edge_index,
        torch.tensor([[0], [0]]),
    )
    assert torch.equal(
        record.graph.edges[("paper", "written_by", "author")].edge_index,
        torch.tensor([[0], [0]]),
    )
    assert record.src_index == 0
    assert record.dst_index == 0


def test_link_neighbor_sampler_supports_mixed_hetero_edge_types_from_base_sampler():
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

    class MixedEdgeTypeBaseSampler:
        def sample(self, item):
            return [
                LinkPredictionRecord(
                    graph=item.graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    edge_type=writes,
                ),
                LinkPredictionRecord(
                    graph=item.graph,
                    src_index=2,
                    dst_index=3,
                    label=0,
                    edge_type=cites,
                ),
            ]

    sampler = LinkNeighborSampler(num_neighbors=[-1], base_sampler=MixedEdgeTypeBaseSampler())
    records = sampler.sample(LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, edge_type=writes))

    assert len(records) == 2
    assert records[0].graph is records[1].graph
    assert records[0].edge_type == writes
    assert records[1].edge_type == cites
    assert all(int(record.src_index) >= 0 for record in records)
    assert all(int(record.dst_index) >= 0 for record in records)


def test_link_neighbor_sampler_output_blocks_build_relation_local_hetero_blocks():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]], dtype=torch.long),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]], dtype=torch.long),
                "edge_weight": torch.tensor([100.0, 110.0, 120.0]),
            },
        },
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1, -1], output_blocks=True)

    record = sampler.sample(
        LinkPredictionRecord(
            graph=graph,
            src_index=0,
            dst_index=0,
            label=1,
            edge_type=WRITES,
        )
    )

    assert record.blocks is not None
    assert len(record.blocks) == 2
    outer_block, inner_block = record.blocks
    assert outer_block.edge_type == WRITES
    assert outer_block.src_type == "author"
    assert outer_block.dst_type == "paper"
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(outer_block.dstdata["x"].view(-1), torch.tensor([1.0, 2.0]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([0], dtype=torch.long))
    assert torch.equal(inner_block.src_n_id, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))
    assert torch.equal(inner_block.dstdata["x"].view(-1), torch.tensor([1.0]))


def test_link_neighbor_sampler_output_blocks_materialize_multi_relation_hetero_blocks_from_mixed_edge_types():
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            WRITES: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            WRITTEN_BY: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
            cites: {"edge_index": torch.tensor([[0, 2], [2, 3]])},
        },
    )

    class MixedEdgeTypeBaseSampler:
        def sample(self, item):
            return [
                LinkPredictionRecord(
                    graph=item.graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    edge_type=WRITES,
                ),
                LinkPredictionRecord(
                    graph=item.graph,
                    src_index=2,
                    dst_index=3,
                    label=0,
                    edge_type=cites,
                ),
            ]

    sampler = LinkNeighborSampler(
        num_neighbors=[-1],
        base_sampler=MixedEdgeTypeBaseSampler(),
        output_blocks=True,
    )

    records = sampler.sample(LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, edge_type=WRITES))

    assert len(records) == 2
    assert records[0].blocks is not None
    assert records[1].blocks is not None
    assert len(records[0].blocks) == 1
    block = records[0].blocks[0]
    assert isinstance(block, HeteroBlock)
    assert block.edge_types == (WRITES, WRITTEN_BY, cites)
    assert torch.equal(block.dst_n_id["author"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(block.dst_n_id["paper"], torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(block.edata(WRITES)["e_id"], torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(block.edata(WRITTEN_BY)["e_id"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(block.edata(cites)["e_id"], torch.tensor([0, 1], dtype=torch.long))


def test_loader_routes_link_neighbor_sampler_through_plan_execution():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.randn(5, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
        ]
    )

    class PlanOnlyLinkNeighborSampler(LinkNeighborSampler):
        def sample(self, item):
            raise AssertionError("loader should use build_plan instead of the legacy sample path")

    loader = Loader(
        dataset=dataset,
        sampler=PlanOnlyLinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=UniformNegativeLinkSampler(num_negatives=1),
        ),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))
    assert torch.equal(batch.query_index, torch.tensor([0, 0]))
    assert batch.graph.x.size(0) == 4

def test_link_neighbor_sampler_prefetch_option_materializes_homo_features_into_record_graph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.zeros(4, 2),
        edge_data={"edge_weight": torch.zeros(3)},
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])),
            ("edge", HOMO_EDGE, "edge_weight"): InMemoryTensorStore(torch.tensor([10.0, 20.0, 30.0])),
        }
    )
    graph.feature_store = feature_store
    sampler = LinkNeighborSampler(
        num_neighbors=[-1],
        node_feature_names=("x",),
        edge_feature_names=("edge_weight",),
    )

    record = sampler.sample(LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1))

    assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(record.graph.x, torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]))
    assert torch.equal(record.graph.edata["edge_weight"], torch.tensor([10.0, 20.0, 30.0]))


def test_link_neighbor_sampler_prefetch_option_materializes_hetero_features_into_batch_graph():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.zeros(2, 2)},
            "paper": {"x": torch.zeros(3, 2)},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "edge_weight": torch.zeros(2),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[1, 2], [0, 1]]),
                "edge_weight": torch.zeros(2),
            },
        },
    )
    feature_store = FeatureStore(
        {
            ("node", "paper", "x"): InMemoryTensorStore(torch.tensor([[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]])),
            ("edge", WRITES, "edge_weight"): InMemoryTensorStore(torch.tensor([11.0, 13.0])),
        }
    )
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    edge_type=WRITES,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"paper": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=feature_store,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([1]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[6.0, 0.0]]))
    assert torch.equal(batch.graph.edges[WRITES].edge_weight, torch.tensor([11.0]))




def test_link_neighbor_sampler_prefetch_option_keeps_sampled_shard_global_ids_aligned_through_coordinator(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2], [1, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[1].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([2, 3]))
    assert torch.equal(batch.graph.x, torch.tensor([[2.0], [3.0]]))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([20.0]))


def test_link_neighbor_sampler_prefetch_option_keeps_sampled_shard_global_ids_aligned_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2], [1, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[1].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([2, 3]))
    assert torch.equal(batch.graph.x, torch.tensor([[2.0], [3.0]]))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([20.0]))



def test_link_neighbor_sampler_stitched_link_sampling_crosses_partition_boundaries_through_coordinator(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(batch.graph.edata["e_id"], torch.tensor([0, 1]))
    assert torch.equal(batch.graph.x, torch.tensor([[0.0], [1.0], [2.0]]))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))
    assert torch.equal(batch.labels, torch.tensor([1.0]))


def test_link_neighbor_sampler_stitched_link_sampling_crosses_partition_boundaries_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(batch.graph.edata["e_id"], torch.tensor([0, 1]))
    assert torch.equal(batch.graph.x, torch.tensor([[0.0], [1.0], [2.0]]))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))
    assert torch.equal(batch.labels, torch.tensor([1.0]))



def test_link_neighbor_sampler_stitched_output_blocks_materialize_blocks_through_coordinator(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0], dtype=torch.float32)},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    assert torch.equal(batch.graph.edata["e_id"], torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(batch.graph.x.view(-1), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([10.0, 20.0]))
    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block, inner_block = batch.blocks
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 20.0]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(inner_block.src_n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0]))


def test_link_neighbor_sampler_stitched_output_blocks_materialize_blocks_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0], dtype=torch.float32)},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    assert torch.equal(batch.graph.edata["e_id"], torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(batch.graph.x.view(-1), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([10.0, 20.0]))
    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block, inner_block = batch.blocks
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 20.0]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(inner_block.src_n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0]))



def test_link_neighbor_sampler_stitched_output_blocks_exclude_seed_edges_from_message_passing_blocks(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [0, 1, 2]], dtype=torch.long),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    metadata={"exclude_seed_edges": True},
                )
            ]
        ),
        sampler=LinkNeighborSampler(num_neighbors=[-1], output_blocks=True),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.blocks is not None
    assert len(batch.blocks) == 1
    assert torch.equal(batch.graph.edata["e_id"], torch.tensor([0, 2], dtype=torch.long))
    assert 1 not in set(batch.blocks[0].edata["e_id"].tolist())
    assert torch.equal(batch.blocks[0].edata["e_id"], torch.tensor([0], dtype=torch.long))


def test_link_neighbor_sampler_stitched_output_blocks_exclude_seed_edges_from_message_passing_blocks_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [0, 1, 2]], dtype=torch.long),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    metadata={"exclude_seed_edges": True},
                )
            ]
        ),
        sampler=LinkNeighborSampler(num_neighbors=[-1], output_blocks=True),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.blocks is not None
    assert len(batch.blocks) == 1
    assert torch.equal(batch.graph.edata["e_id"], torch.tensor([0, 2], dtype=torch.long))
    assert 1 not in set(batch.blocks[0].edata["e_id"].tolist())
    assert torch.equal(batch.blocks[0].edata["e_id"], torch.tensor([0], dtype=torch.long))



def test_link_neighbor_sampler_stitched_hetero_link_sampling_crosses_partition_boundaries_through_coordinator(tmp_path):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 2], [0, 0]]),
                "edge_weight": torch.tensor([10.0, 20.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 0], [0, 2]]),
                "edge_weight": torch.tensor([100.0, 200.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    label=1,
                    edge_type=WRITES,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",), WRITTEN_BY: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.edge_type == WRITES
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0]))
    assert torch.equal(batch.graph.nodes["author"].x, torch.tensor([[10.0], [30.0]]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[1.0]]))
    assert torch.equal(
        batch.graph.edges[WRITES].edge_index,
        torch.tensor([[0, 1], [0, 0]]),
    )
    assert torch.equal(batch.graph.edges[WRITES].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[WRITES].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(
        batch.graph.edges[WRITTEN_BY].edge_index,
        torch.tensor([[0, 0], [0, 1]]),
    )
    assert torch.equal(batch.graph.edges[WRITTEN_BY].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[WRITTEN_BY].edge_weight, torch.tensor([100.0, 200.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([0]))
    assert torch.equal(batch.labels, torch.tensor([1.0]))


def test_link_neighbor_sampler_stitched_hetero_link_sampling_crosses_partition_boundaries_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 2], [0, 0]]),
                "edge_weight": torch.tensor([10.0, 20.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 0], [0, 2]]),
                "edge_weight": torch.tensor([100.0, 200.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    label=1,
                    edge_type=WRITES,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",), WRITTEN_BY: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.edge_type == WRITES
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0]))
    assert torch.equal(batch.graph.nodes["author"].x, torch.tensor([[10.0], [30.0]]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[1.0]]))
    assert torch.equal(
        batch.graph.edges[WRITES].edge_index,
        torch.tensor([[0, 1], [0, 0]]),
    )
    assert torch.equal(batch.graph.edges[WRITES].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[WRITES].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(
        batch.graph.edges[WRITTEN_BY].edge_index,
        torch.tensor([[0, 0], [0, 1]]),
    )
    assert torch.equal(batch.graph.edges[WRITTEN_BY].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[WRITTEN_BY].edge_weight, torch.tensor([100.0, 200.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([0]))
    assert torch.equal(batch.labels, torch.tensor([1.0]))


def test_link_neighbor_sampler_stitched_hetero_output_blocks_materialize_relation_local_blocks_through_coordinator(tmp_path):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]], dtype=torch.long),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]], dtype=torch.long),
                "edge_weight": torch.tensor([100.0, 110.0, 120.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    label=1,
                    edge_type=WRITES,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",), WRITTEN_BY: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block, inner_block = batch.blocks
    assert batch.edge_type == WRITES
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(batch.graph.nodes["author"].x.view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(batch.graph.nodes["paper"].x.view(-1), torch.tensor([1.0, 2.0]))
    assert outer_block.edge_type == WRITES
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(outer_block.dstdata["x"].view(-1), torch.tensor([1.0, 2.0]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([0], dtype=torch.long))
    assert torch.equal(inner_block.src_n_id, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))


def test_link_neighbor_sampler_stitched_hetero_output_blocks_materialize_relation_local_blocks_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]], dtype=torch.long),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]], dtype=torch.long),
                "edge_weight": torch.tensor([100.0, 110.0, 120.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    label=1,
                    edge_type=WRITES,
                )
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",), WRITTEN_BY: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block, inner_block = batch.blocks
    assert batch.edge_type == WRITES
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(batch.graph.nodes["author"].x.view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(batch.graph.nodes["paper"].x.view(-1), torch.tensor([1.0, 2.0]))
    assert outer_block.edge_type == WRITES
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(outer_block.dstdata["x"].view(-1), torch.tensor([1.0, 2.0]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([0], dtype=torch.long))
    assert torch.equal(inner_block.src_n_id, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))


def test_link_neighbor_sampler_stitched_hetero_output_blocks_materialize_multi_relation_blocks_through_coordinator(
    tmp_path,
):
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0], [4.0]])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                "edge_weight": torch.tensor([10.0, 20.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[1, 2], [0, 1]], dtype=torch.long),
                "edge_weight": torch.tensor([100.0, 200.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 2], [2, 3]], dtype=torch.long),
                "edge_weight": torch.tensor([1000.0, 2000.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=1)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
    }

    class MixedEdgeTypeBaseSampler:
        def sample(self, item):
            return [
                LinkPredictionRecord(
                    graph=item.graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    edge_type=WRITES,
                ),
                LinkPredictionRecord(
                    graph=item.graph,
                    src_index=2,
                    dst_index=3,
                    label=0,
                    edge_type=cites,
                ),
            ]

    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [LinkPredictionRecord(graph=shards[0].graph, src_index=0, dst_index=1, label=1, edge_type=WRITES)]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=MixedEdgeTypeBaseSampler(),
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={
                WRITES: ("edge_weight",),
                WRITTEN_BY: ("edge_weight",),
                cites: ("edge_weight",),
            },
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.blocks is not None
    assert len(batch.blocks) == 1
    block = batch.blocks[0]
    assert isinstance(block, HeteroBlock)
    assert block.edge_types == (WRITES, WRITTEN_BY, cites)
    assert torch.equal(block.dst_n_id["author"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(block.dst_n_id["paper"], torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(block.edata(WRITES)["edge_weight"], torch.tensor([10.0, 20.0]))
    assert torch.equal(block.edata(WRITTEN_BY)["edge_weight"], torch.tensor([100.0]))
    assert torch.equal(block.edata(cites)["edge_weight"], torch.tensor([1000.0, 2000.0]))


def test_link_neighbor_sampler_stitched_hetero_output_blocks_materialize_multi_relation_blocks_through_store_backed_coordinator(
    tmp_path,
):
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0], [4.0]])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                "edge_weight": torch.tensor([10.0, 20.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[1, 2], [0, 1]], dtype=torch.long),
                "edge_weight": torch.tensor([100.0, 200.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 2], [2, 3]], dtype=torch.long),
                "edge_weight": torch.tensor([1000.0, 2000.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=1)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
    }

    class MixedEdgeTypeBaseSampler:
        def sample(self, item):
            return [
                LinkPredictionRecord(
                    graph=item.graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    edge_type=WRITES,
                ),
                LinkPredictionRecord(
                    graph=item.graph,
                    src_index=2,
                    dst_index=3,
                    label=0,
                    edge_type=cites,
                ),
            ]

    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset(
            [LinkPredictionRecord(graph=shards[0].graph, src_index=0, dst_index=1, label=1, edge_type=WRITES)]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=MixedEdgeTypeBaseSampler(),
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={
                WRITES: ("edge_weight",),
                WRITTEN_BY: ("edge_weight",),
                cites: ("edge_weight",),
            },
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.blocks is not None
    assert len(batch.blocks) == 1
    block = batch.blocks[0]
    assert isinstance(block, HeteroBlock)
    assert block.edge_types == (WRITES, WRITTEN_BY, cites)
    assert torch.equal(block.dst_n_id["author"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(block.dst_n_id["paper"], torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(block.edata(WRITES)["edge_weight"], torch.tensor([10.0, 20.0]))
    assert torch.equal(block.edata(WRITTEN_BY)["edge_weight"], torch.tensor([100.0]))
    assert torch.equal(block.edata(cites)["edge_weight"], torch.tensor([1000.0, 2000.0]))


def test_link_neighbor_sampler_output_blocks_preserve_order_and_global_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3, 2], [1, 2, 2, 3]], dtype=torch.long),
        x=torch.arange(16, dtype=torch.float32).view(4, 4),
        n_id=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        edge_data={"e_id": torch.tensor([100, 101, 102, 103], dtype=torch.long)},
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1, -1], output_blocks=True)

    record = sampler.sample(LinkPredictionRecord(graph=graph, src_index=1, dst_index=2, label=1))

    assert record.blocks is not None
    assert len(record.blocks) == 2
    outer_block, inner_block = record.blocks
    assert torch.equal(outer_block.dst_n_id, torch.tensor([10, 11, 12, 13], dtype=torch.long))
    assert torch.equal(outer_block.src_n_id, torch.tensor([10, 11, 12, 13], dtype=torch.long))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([100, 101, 102, 103], dtype=torch.long))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([11, 12], dtype=torch.long))
    assert torch.equal(inner_block.src_n_id, torch.tensor([11, 12, 10, 13], dtype=torch.long))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([100, 101, 102], dtype=torch.long))



def test_link_neighbor_sampler_output_blocks_use_only_sampled_edges():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 4, 1], [1, 1, 1, 1, 5]], dtype=torch.long),
        x=torch.randn(6, 2),
        edge_data={"e_id": torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)},
    )
    sampler = LinkNeighborSampler(num_neighbors=[2], seed=0, output_blocks=True)

    record = sampler.sample(LinkPredictionRecord(graph=graph, src_index=1, dst_index=5, label=1))

    assert record.blocks is not None
    assert len(record.blocks) == 1
    sampled_node_ids = set(record.graph.n_id.tolist())
    omitted_edge_ids = [
        edge_id
        for edge_id, (src_index, dst_index) in zip(graph.edata["e_id"].tolist(), graph.edge_index.t().tolist())
        if int(dst_index) in {1, 5} and int(src_index) not in sampled_node_ids
    ]
    assert omitted_edge_ids
    assert set(record.blocks[0].edata["e_id"].tolist()).isdisjoint(omitted_edge_ids)



def test_link_neighbor_sampler_output_blocks_exclude_seed_edges_from_message_passing_blocks():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        x=torch.randn(3, 2),
        edge_data={"e_id": torch.tensor([100, 101], dtype=torch.long)},
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1], output_blocks=True)

    record = sampler.sample(
        LinkPredictionRecord(
            graph=graph,
            src_index=1,
            dst_index=2,
            label=1,
            metadata={"exclude_seed_edges": True},
        )
    )

    assert record.blocks is not None
    assert torch.equal(record.graph.edata["e_id"], torch.tensor([100, 101], dtype=torch.long))
    assert torch.equal(record.blocks[0].edata["e_id"], torch.tensor([100], dtype=torch.long))



def test_loader_materializes_link_prediction_batch_blocks_for_homogeneous_sampling():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3, 2], [1, 2, 2, 3]], dtype=torch.long),
        x=torch.arange(16, dtype=torch.float32).view(4, 4),
        n_id=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        edge_data={"e_id": torch.tensor([100, 101, 102, 103], dtype=torch.long)},
    )
    loader = Loader(
        dataset=ListDataset([LinkPredictionRecord(graph=graph, src_index=1, dst_index=2, label=1)]),
        sampler=LinkNeighborSampler(num_neighbors=[-1, -1], output_blocks=True),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    assert torch.equal(batch.blocks[0].dst_n_id, torch.tensor([10, 11, 12, 13], dtype=torch.long))
    assert torch.equal(batch.blocks[1].dst_n_id, torch.tensor([11, 12], dtype=torch.long))



def test_link_neighbor_sampler_output_blocks_keep_fixed_hop_count_for_hetero_sampling():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(3, 4)},
            "paper": {"x": torch.randn(2, 4)},
        },
        edges={
            WRITES: {"edge_index": torch.tensor([[0, 2], [0, 0]], dtype=torch.long)},
            WRITTEN_BY: {"edge_index": torch.tensor([[0, 0], [0, 2]], dtype=torch.long)},
        },
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1, -1], output_blocks=True)

    record = sampler.sample(
        LinkPredictionRecord(
            graph=graph,
            src_index=0,
            dst_index=0,
            label=1,
            edge_type=WRITES,
        )
    )

    assert record.blocks is not None
    assert len(record.blocks) == 2
    assert torch.equal(record.blocks[0].dst_n_id, torch.tensor([0], dtype=torch.long))
    assert torch.equal(record.blocks[1].dst_n_id, torch.tensor([0], dtype=torch.long))
