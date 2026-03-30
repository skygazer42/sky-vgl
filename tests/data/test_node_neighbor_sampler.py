import torch

from vgl import Graph, HeteroBlock
from vgl.core.batch import NodeBatch
from vgl.data.loader import Loader
from vgl.data.dataset import ListDataset
from vgl.data.sampler import NodeNeighborSampler
from vgl.dataloading.plan import PlanStage
from vgl.distributed import LocalGraphShard, LocalSamplingCoordinator, write_partitioned_graph
from vgl.distributed.coordinator import StoreBackedSamplingCoordinator
from vgl.storage import FeatureStore, InMemoryTensorStore


def test_node_neighbor_sampler_extracts_local_subgraph_and_seed_index():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
        y=torch.tensor([0, 1, 0, 1]),
    )
    sampler = NodeNeighborSampler(num_neighbors=[-1])

    sample = sampler.sample((graph, {"seed": 1, "sample_id": "n1"}))

    assert sample.sample_id == "n1"
    assert sample.subgraph_seed == 1
    assert torch.equal(sample.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(sample.graph.edge_index, torch.tensor([[0, 1, 3], [1, 2, 1]]))


def test_loader_builds_node_batch_from_node_neighbor_samples():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.randn(5, 4),
        y=torch.tensor([0, 1, 0, 1, 0]),
    )
    dataset = ListDataset(
        [
            (graph, {"seed": 0, "sample_id": "n0"}),
            (graph, {"seed": 2, "sample_id": "n2"}),
        ]
    )
    loader = Loader(dataset=dataset, sampler=NodeNeighborSampler(num_neighbors=[-1]), batch_size=2)

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.graph.x.size(0) == 5
    assert torch.equal(batch.seed_index, torch.tensor([0, 3]))
    assert batch.metadata == [{"seed": 0, "sample_id": "n0"}, {"seed": 2, "sample_id": "n2"}]


def test_node_neighbor_sampler_extracts_hetero_local_subgraph():
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.randn(3, 4),
                "y": torch.tensor([0, 1, 0]),
            },
            "author": {
                "x": torch.randn(2, 4),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 2], [0, 1]]),
            },
        },
    )
    sampler = NodeNeighborSampler(num_neighbors=[-1])

    sample = sampler.sample((graph, {"seed": 1, "node_type": "paper", "sample_id": "p1"}))

    assert sample.sample_id == "p1"
    assert sample.metadata["node_type"] == "paper"
    assert sample.subgraph_seed == 0
    assert torch.equal(sample.graph.nodes["paper"].data["n_id"], torch.tensor([1]))
    assert torch.equal(sample.graph.nodes["author"].data["n_id"], torch.tensor([0]))


def _store_backed_coordinator(shards):
    root = next(iter(shards.values())).root
    return StoreBackedSamplingCoordinator.from_partition_dir(root)


def test_loader_routes_node_neighbor_sampler_through_plan_execution():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.randn(5, 4),
        y=torch.tensor([0, 1, 0, 1, 0]),
    )
    dataset = ListDataset(
        [
            (graph, {"seed": 0, "sample_id": "n0"}),
            (graph, {"seed": 2, "sample_id": "n2"}),
        ]
    )

    class PlanOnlyNodeNeighborSampler(NodeNeighborSampler):
        def sample(self, item):
            raise AssertionError("loader should use build_plan instead of the legacy sample path")

    loader = Loader(dataset=dataset, sampler=PlanOnlyNodeNeighborSampler(num_neighbors=[-1]), batch_size=2)

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.graph.x.size(0) == 5
    assert torch.equal(batch.seed_index, torch.tensor([0, 3]))
    assert batch.metadata == [{"seed": 0, "sample_id": "n0"}, {"seed": 2, "sample_id": "n2"}]

HOMO_EDGE = ("node", "to", "node")
WRITES = ("author", "writes", "paper")
WRITTEN_BY = ("paper", "written_by", "author")
CITES = ("paper", "cites", "paper")


class FeaturePlanNodeNeighborSampler(NodeNeighborSampler):
    def __init__(self, num_neighbors, *, stages):
        super().__init__(num_neighbors=num_neighbors)
        self._stages = tuple(stages)

    def build_plan(self, item):
        return super().build_plan(item).append(*self._stages)


def test_node_neighbor_sampler_materializes_fetched_homo_features_into_subgraph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.zeros(4, 2),
        y=torch.tensor([0, 1, 0, 1]),
        edge_data={"edge_weight": torch.zeros(3)},
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])),
            ("edge", HOMO_EDGE, "edge_weight"): InMemoryTensorStore(torch.tensor([10.0, 20.0, 30.0])),
        }
    )
    sampler = FeaturePlanNodeNeighborSampler(
        num_neighbors=[-1],
        stages=(
            PlanStage(
                "fetch_node_features",
                params={
                    "node_type": "node",
                    "feature_names": ("x",),
                    "index_key": "node_ids",
                    "output_key": "node_features",
                },
            ),
            PlanStage(
                "fetch_edge_features",
                params={
                    "edge_type": HOMO_EDGE,
                    "feature_names": ("edge_weight",),
                    "index_key": "edge_ids",
                    "output_key": "edge_features",
                },
            ),
        ),
    )
    loader = Loader(
        dataset=ListDataset([(graph, {"seed": 1, "sample_id": "n1"})]),
        sampler=sampler,
        batch_size=1,
        feature_store=feature_store,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(batch.graph.x, torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([10.0, 20.0, 30.0]))


def test_node_neighbor_sampler_materializes_fetched_hetero_features_into_subgraph():
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.zeros(3, 2),
                "y": torch.tensor([0, 1, 0]),
            },
            "author": {
                "x": torch.zeros(2, 2),
            },
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
    sampler = FeaturePlanNodeNeighborSampler(
        num_neighbors=[-1],
        stages=(
            PlanStage(
                "fetch_node_features",
                params={
                    "node_type": "paper",
                    "feature_names": ("x",),
                    "index_key": "node_ids_by_type",
                    "output_key": "paper_features",
                },
            ),
            PlanStage(
                "fetch_edge_features",
                params={
                    "edge_type": WRITES,
                    "feature_names": ("edge_weight",),
                    "index_key": "edge_ids_by_type",
                    "output_key": "writes_features",
                },
            ),
        ),
    )
    loader = Loader(
        dataset=ListDataset([(graph, {"seed": 1, "node_type": "paper", "sample_id": "p1"})]),
        sampler=sampler,
        batch_size=1,
        feature_store=feature_store,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.graph.nodes["paper"].data["n_id"], torch.tensor([1]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[6.0, 0.0]]))
    assert torch.equal(batch.graph.edges[WRITES].edge_weight, torch.tensor([11.0]))

def test_node_neighbor_sampler_prefetch_option_appends_homo_feature_stages():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.zeros(4, 2),
        y=torch.tensor([0, 1, 0, 1]),
        edge_data={"edge_weight": torch.zeros(3)},
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])),
            ("edge", HOMO_EDGE, "edge_weight"): InMemoryTensorStore(torch.tensor([10.0, 20.0, 30.0])),
        }
    )
    loader = Loader(
        dataset=ListDataset([(graph, {"seed": 1, "sample_id": "n1"})]),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=feature_store,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.x, torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([10.0, 20.0, 30.0]))


def test_node_neighbor_sampler_prefetch_option_appends_hetero_feature_stages():
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.zeros(3, 2),
                "y": torch.tensor([0, 1, 0]),
            },
            "author": {
                "x": torch.zeros(2, 2),
            },
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
        dataset=ListDataset([(graph, {"seed": 1, "node_type": "paper", "sample_id": "p1"})]),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"paper": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=feature_store,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[6.0, 0.0]]))
    assert torch.equal(batch.graph.edges[WRITES].edge_weight, torch.tensor([11.0]))




def test_node_neighbor_sampler_extracts_shared_subgraph_for_multiple_homo_multi_seed_contexts():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
        y=torch.tensor([0, 1, 0, 1]),
    )
    sampler = NodeNeighborSampler(num_neighbors=[-1])

    samples = sampler.sample((graph, {"seed": [1, 3], "sample_id": "n13"}))

    assert isinstance(samples, list)
    assert len(samples) == 2
    assert samples[0].graph is samples[1].graph
    assert torch.equal(samples[0].graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert [sample.sample_id for sample in samples] == ["n13", "n13"]
    assert [sample.subgraph_seed for sample in samples] == [1, 3]
    assert [sample.metadata["seed"] for sample in samples] == [1, 3]


def test_loader_builds_node_batch_from_multi_seed_node_neighbor_context():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.randn(4, 4),
        y=torch.tensor([0, 1, 0, 1]),
    )
    loader = Loader(
        dataset=ListDataset([(graph, {"seed": [1, 3], "sample_id": "n13"})]),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.graph.x.size(0) == 4
    assert torch.equal(batch.seed_index, torch.tensor([1, 3]))
    assert batch.metadata == [
        {"seed": 1, "sample_id": "n13"},
        {"seed": 3, "sample_id": "n13"},
    ]


def test_loader_builds_hetero_node_batch_from_multi_seed_node_neighbor_context():
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4), "y": torch.tensor([0, 1, 0])},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    loader = Loader(
        dataset=ListDataset([(graph, {"seed": [1, 2], "node_type": "paper", "sample_id": "p12"})]),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([1, 2]))
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 1]))
    assert torch.equal(batch.seed_index, torch.tensor([0, 1]))
    assert batch.metadata == [
        {"seed": 1, "node_type": "paper", "sample_id": "p12"},
        {"seed": 2, "node_type": "paper", "sample_id": "p12"},
    ]



def test_node_neighbor_sampler_prefetch_option_aligns_partition_shard_features_through_coordinator(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2], [1, 3]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
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
                (shards[0].graph, {"seed": 0, "sample_id": "part0"}),
                (shards[1].graph, {"seed": 1, "sample_id": "part1"}),
            ]
        ),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=2,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(batch.graph.x, torch.arange(4, dtype=torch.float32).view(4, 1))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([10.0, 20.0]))



def test_node_neighbor_sampler_stitched_hetero_sampling_crosses_partition_boundaries_through_coordinator(tmp_path):
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                "y": torch.tensor([0, 1, 0, 1]),
            },
            "author": {
                "x": torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
            },
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 2], [0, 1]]),
                "edge_weight": torch.tensor([10.0, 20.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 1], [0, 2]]),
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
            [(shards[0].graph, {"seed": 1, "node_type": "paper", "sample_id": "stitched_hetero"})]
        ),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"paper": ("x",), "author": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",), WRITTEN_BY: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([1]))
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([2]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[2.0]]))
    assert torch.equal(batch.graph.nodes["author"].x, torch.tensor([[30.0]]))
    assert torch.equal(batch.graph.edges[WRITES].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(batch.graph.edges[WRITES].e_id, torch.tensor([1]))
    assert torch.equal(batch.graph.edges[WRITES].edge_weight, torch.tensor([20.0]))
    assert torch.equal(batch.graph.edges[WRITTEN_BY].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(batch.graph.edges[WRITTEN_BY].e_id, torch.tensor([1]))
    assert torch.equal(batch.graph.edges[WRITTEN_BY].edge_weight, torch.tensor([200.0]))
    assert torch.equal(batch.seed_index, torch.tensor([0]))
    assert batch.metadata == [{"seed": 1, "node_type": "paper", "sample_id": "stitched_hetero"}]


def test_node_neighbor_sampler_stitched_hetero_sampling_crosses_partition_boundaries_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                "y": torch.tensor([0, 1, 0, 1]),
            },
            "author": {
                "x": torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
            },
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 2], [0, 1]]),
                "edge_weight": torch.tensor([10.0, 20.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 1], [0, 2]]),
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
            [(shards[0].graph, {"seed": 1, "node_type": "paper", "sample_id": "stitched_hetero_store"})]
        ),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"paper": ("x",), "author": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",), WRITTEN_BY: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([1]))
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([2]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[2.0]]))
    assert torch.equal(batch.graph.nodes["author"].x, torch.tensor([[30.0]]))
    assert torch.equal(batch.graph.edges[WRITES].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(batch.graph.edges[WRITES].e_id, torch.tensor([1]))
    assert torch.equal(batch.graph.edges[WRITES].edge_weight, torch.tensor([20.0]))
    assert torch.equal(batch.graph.edges[WRITTEN_BY].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(batch.graph.edges[WRITTEN_BY].e_id, torch.tensor([1]))
    assert torch.equal(batch.graph.edges[WRITTEN_BY].edge_weight, torch.tensor([200.0]))
    assert torch.equal(batch.seed_index, torch.tensor([0]))
    assert batch.metadata == [{"seed": 1, "node_type": "paper", "sample_id": "stitched_hetero_store"}]



def test_node_neighbor_sampler_hetero_output_blocks_materialize_relation_local_blocks():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]]), "y": torch.tensor([0, 1])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]]),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]]),
                "edge_weight": torch.tensor([100.0, 110.0, 120.0]),
            },
        },
    )
    sampler = NodeNeighborSampler(num_neighbors=[-1, -1, -1], output_blocks=True)

    sample = sampler.sample((graph, {"seed": 0, "node_type": "paper", "sample_id": "p0"}))

    assert sample.blocks is not None
    assert len(sample.blocks) == 3
    outer_block, middle_block, inner_block = sample.blocks
    assert torch.equal(sample.graph.nodes["author"].n_id, torch.tensor([0, 2]))
    assert torch.equal(sample.graph.nodes["paper"].n_id, torch.tensor([0, 1]))
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1]))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 2]))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 2]))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(outer_block.dstdata["x"].view(-1), torch.tensor([1.0, 2.0]))
    assert torch.equal(middle_block.dst_n_id, torch.tensor([0]))
    assert torch.equal(middle_block.src_n_id, torch.tensor([0, 2]))
    assert torch.equal(middle_block.edata["e_id"], torch.tensor([0, 2]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([0]))
    assert torch.equal(inner_block.src_n_id, torch.tensor([0, 2]))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0, 2]))


def test_node_neighbor_sampler_stitched_hetero_output_blocks_materialize_relation_local_blocks_through_coordinator(
    tmp_path,
):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]]), "y": torch.tensor([0, 1])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]]),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]]),
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
            [(shards[0].graph, {"seed": 0, "node_type": "paper", "sample_id": "stitched_hetero_blocks"})]
        ),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",), WRITTEN_BY: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.blocks is not None
    assert len(batch.blocks) == 3
    outer_block, middle_block, inner_block = batch.blocks
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.nodes["author"].x.view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(batch.graph.nodes["paper"].x.view(-1), torch.tensor([1.0, 2.0]))
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1]))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 2]))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 2]))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(outer_block.dstdata["x"].view(-1), torch.tensor([1.0, 2.0]))
    assert torch.equal(middle_block.dst_n_id, torch.tensor([0]))
    assert torch.equal(middle_block.src_n_id, torch.tensor([0, 2]))
    assert torch.equal(middle_block.edata["e_id"], torch.tensor([0, 2]))
    assert torch.equal(middle_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([0]))
    assert torch.equal(inner_block.src_n_id, torch.tensor([0, 2]))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0, 2]))
    assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))
    assert torch.equal(batch.seed_index, torch.tensor([0]))
    assert batch.metadata == [{"seed": 0, "node_type": "paper", "sample_id": "stitched_hetero_blocks"}]


def test_node_neighbor_sampler_stitched_hetero_output_blocks_materialize_relation_local_blocks_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]]), "y": torch.tensor([0, 1])},
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 0, 2], [0, 1, 0]]),
                "edge_weight": torch.tensor([10.0, 11.0, 12.0]),
            },
            WRITTEN_BY: {
                "edge_index": torch.tensor([[0, 1, 0], [0, 0, 2]]),
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
                (
                    shards[0].graph,
                    {"seed": 0, "node_type": "paper", "sample_id": "stitched_hetero_blocks_store"},
                )
            ]
        ),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",), WRITTEN_BY: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.blocks is not None
    assert len(batch.blocks) == 3
    outer_block, middle_block, inner_block = batch.blocks
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.nodes["author"].x.view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(batch.graph.nodes["paper"].x.view(-1), torch.tensor([1.0, 2.0]))
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1]))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 2]))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 2]))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([10.0, 30.0]))
    assert torch.equal(outer_block.dstdata["x"].view(-1), torch.tensor([1.0, 2.0]))
    assert torch.equal(middle_block.dst_n_id, torch.tensor([0]))
    assert torch.equal(middle_block.src_n_id, torch.tensor([0, 2]))
    assert torch.equal(middle_block.edata["e_id"], torch.tensor([0, 2]))
    assert torch.equal(middle_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([0]))
    assert torch.equal(inner_block.src_n_id, torch.tensor([0, 2]))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0, 2]))
    assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0, 12.0]))
    assert torch.equal(batch.seed_index, torch.tensor([0]))
    assert batch.metadata == [{"seed": 0, "node_type": "paper", "sample_id": "stitched_hetero_blocks_store"}]


def test_node_neighbor_sampler_stitched_sampling_crosses_partition_boundaries_through_coordinator(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset([(shards[0].graph, {"seed": 1, "sample_id": "stitched"})]),
        sampler=NodeNeighborSampler(num_neighbors=[-1, -1]),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]))
    assert torch.equal(batch.graph.edata["e_id"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(batch.graph.x, torch.arange(4, dtype=torch.float32).view(4, 1))
    assert torch.equal(batch.seed_index, torch.tensor([1]))
    assert batch.metadata == [{"seed": 1, "sample_id": "stitched"}]


def test_node_neighbor_sampler_stitched_sampling_materializes_blocks_through_coordinator(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset([(shards[0].graph, {"seed": 1, "sample_id": "stitched_blocks"})]),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block, inner_block = batch.blocks
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 3]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 20.0, 40.0]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([1]))
    assert torch.equal(inner_block.src_n_id, torch.tensor([1, 0]))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0]))
    assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0]))


def test_node_neighbor_sampler_stitched_sampling_materializes_blocks_through_store_backed_coordinator(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(4, dtype=torch.float32).view(4, 1),
        y=torch.tensor([0, 1, 0, 1]),
        edge_data={"edge_weight": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = _store_backed_coordinator(shards)
    loader = Loader(
        dataset=ListDataset([(shards[0].graph, {"seed": 1, "sample_id": "stitched_blocks_store"})]),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block, inner_block = batch.blocks
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(outer_block.dst_n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(outer_block.src_n_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([0, 1, 3]))
    assert torch.equal(outer_block.srcdata["x"].view(-1), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(outer_block.edata["edge_weight"], torch.tensor([10.0, 20.0, 40.0]))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([1]))
    assert torch.equal(inner_block.src_n_id, torch.tensor([1, 0]))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([0]))
    assert torch.equal(inner_block.edata["edge_weight"], torch.tensor([10.0]))
    assert torch.equal(batch.seed_index, torch.tensor([1]))
    assert batch.metadata == [{"seed": 1, "sample_id": "stitched_blocks_store"}]



def test_node_neighbor_sampler_stitched_output_blocks_keep_fixed_hop_count_when_frontier_exhausts(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        x=torch.arange(2, dtype=torch.float32).view(2, 1),
        y=torch.tensor([0, 1]),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    loader = Loader(
        dataset=ListDataset([(shards[1].graph, {"seed": 0, "sample_id": "stitched_short"})]),
        sampler=NodeNeighborSampler(num_neighbors=[-1, -1], output_blocks=True),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1]))
    assert torch.equal(batch.blocks[0].dst_n_id, torch.tensor([0, 1]))
    assert torch.equal(batch.blocks[1].dst_n_id, torch.tensor([1]))



def test_node_neighbor_sampler_output_blocks_preserve_order_and_global_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 2]], dtype=torch.long),
        x=torch.arange(16, dtype=torch.float32).view(4, 4),
        n_id=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        edge_data={"e_id": torch.tensor([100, 101, 102], dtype=torch.long)},
    )
    sampler = NodeNeighborSampler(num_neighbors=[-1, -1], output_blocks=True)

    sample = sampler.sample((graph, {"seed": 1, "sample_id": "n1"}))

    assert sample.blocks is not None
    assert len(sample.blocks) == 2
    outer_block, inner_block = sample.blocks
    assert torch.equal(sample.graph.n_id, torch.tensor([10, 11, 12, 13], dtype=torch.long))
    assert torch.equal(outer_block.dst_n_id, torch.tensor([10, 11, 12], dtype=torch.long))
    assert torch.equal(outer_block.src_n_id, torch.tensor([10, 11, 12, 13], dtype=torch.long))
    assert torch.equal(outer_block.edata["e_id"], torch.tensor([100, 101, 102], dtype=torch.long))
    assert torch.equal(inner_block.dst_n_id, torch.tensor([11], dtype=torch.long))
    assert torch.equal(inner_block.src_n_id, torch.tensor([11, 10], dtype=torch.long))
    assert torch.equal(inner_block.edata["e_id"], torch.tensor([100], dtype=torch.long))



def test_node_neighbor_sampler_output_blocks_keep_fixed_hop_count_when_frontier_exhausts():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        x=torch.randn(2, 2),
    )
    sampler = NodeNeighborSampler(num_neighbors=[-1, -1], output_blocks=True)

    sample = sampler.sample((graph, {"seed": 1, "sample_id": "n1"}))

    assert sample.blocks is not None
    assert len(sample.blocks) == 2
    assert torch.equal(sample.blocks[0].dst_n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(sample.blocks[1].dst_n_id, torch.tensor([1], dtype=torch.long))



def test_node_neighbor_sampler_output_blocks_use_only_sampled_edges():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 4, 1], [1, 1, 1, 1, 5]], dtype=torch.long),
        x=torch.randn(6, 2),
        edge_data={"e_id": torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)},
    )
    sampler = NodeNeighborSampler(num_neighbors=[2], seed=0, output_blocks=True)

    sample = sampler.sample((graph, {"seed": 1, "sample_id": "n1"}))

    assert sample.blocks is not None
    assert len(sample.blocks) == 1
    sampled_node_ids = set(sample.graph.n_id.tolist())
    omitted_edge_ids = [
        edge_id
        for edge_id, (src_index, dst_index) in zip(graph.edata["e_id"].tolist(), graph.edge_index.t().tolist())
        if int(dst_index) == 1 and int(src_index) not in sampled_node_ids
    ]
    assert omitted_edge_ids
    assert set(sample.blocks[0].edata["e_id"].tolist()).isdisjoint(omitted_edge_ids)



def test_loader_materializes_node_batch_blocks_for_homogeneous_sampling():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 2]], dtype=torch.long),
        x=torch.arange(16, dtype=torch.float32).view(4, 4),
        n_id=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        edge_data={"e_id": torch.tensor([100, 101, 102], dtype=torch.long)},
    )
    loader = Loader(
        dataset=ListDataset([(graph, {"seed": 1, "sample_id": "n1"})]),
        sampler=NodeNeighborSampler(num_neighbors=[-1, -1], output_blocks=True),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    assert torch.equal(batch.blocks[0].dst_n_id, torch.tensor([10, 11, 12], dtype=torch.long))
    assert torch.equal(batch.blocks[1].dst_n_id, torch.tensor([11], dtype=torch.long))



def test_node_neighbor_sampler_hetero_output_blocks_materialize_multi_relation_hetero_blocks():
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.randn(3, 4),
                "y": torch.tensor([0, 1, 0]),
            },
            "author": {
                "x": torch.randn(2, 4),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            },
            ("paper", "cites", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            },
        },
    )
    sampler = NodeNeighborSampler(num_neighbors=[-1, -1], output_blocks=True)

    sample = sampler.sample((graph, {"seed": 1, "node_type": "paper", "sample_id": "p1"}))

    assert sample.blocks is not None
    assert len(sample.blocks) == 2
    outer_block, inner_block = sample.blocks
    assert isinstance(outer_block, HeteroBlock)
    assert isinstance(inner_block, HeteroBlock)
    assert outer_block.edge_types == (WRITES, CITES)
    assert torch.equal(outer_block.dst_n_id["paper"], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(outer_block.src_n_id["author"], torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(outer_block.src_n_id["paper"], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(inner_block.dst_n_id["paper"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(inner_block.src_n_id["author"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(inner_block.src_n_id["paper"], torch.tensor([1, 0], dtype=torch.long))
    assert torch.equal(inner_block.edata(WRITES)["e_id"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(inner_block.edata(CITES)["e_id"], torch.tensor([0], dtype=torch.long))


def test_node_neighbor_sampler_stitched_hetero_output_blocks_materialize_multi_relation_blocks_through_coordinator(
    tmp_path,
):
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.tensor([[1.0], [2.0], [3.0]]),
                "y": torch.tensor([0, 1, 0]),
            },
            "author": {
                "x": torch.tensor([[10.0], [20.0]]),
            },
        },
        edges={
            WRITES: {
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                "edge_weight": torch.tensor([10.0, 20.0]),
            },
            CITES: {
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
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
            [(shards[0].graph, {"seed": 1, "node_type": "paper", "sample_id": "stitched_multi_relation"})]
        ),
        sampler=NodeNeighborSampler(
            num_neighbors=[-1, -1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={WRITES: ("edge_weight",), CITES: ("edge_weight",)},
            output_blocks=True,
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert isinstance(batch, NodeBatch)
    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block, inner_block = batch.blocks
    assert isinstance(outer_block, HeteroBlock)
    assert isinstance(inner_block, HeteroBlock)
    assert outer_block.edge_types == (WRITES, CITES)
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(outer_block.dst_n_id["paper"], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(outer_block.edata(WRITES)["edge_weight"], torch.tensor([10.0, 20.0]))
    assert torch.equal(outer_block.edata(CITES)["edge_weight"], torch.tensor([100.0, 200.0]))
    assert torch.equal(inner_block.dst_n_id["paper"], torch.tensor([1], dtype=torch.long))
