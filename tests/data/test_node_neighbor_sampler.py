import torch

from vgl import Graph
from vgl.core.batch import NodeBatch
from vgl.data.loader import Loader
from vgl.data.dataset import ListDataset
from vgl.data.sampler import NodeNeighborSampler
from vgl.dataloading.plan import PlanStage
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
    assert torch.equal(sample.graph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0], [0]]))


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

