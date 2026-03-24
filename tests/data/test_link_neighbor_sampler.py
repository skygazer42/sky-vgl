import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import LinkPredictionRecord
from vgl.data.sampler import LinkNeighborSampler, UniformNegativeLinkSampler
from vgl.storage import FeatureStore, InMemoryTensorStore


HOMO_EDGE = ("node", "to", "node")
WRITES = ("author", "writes", "paper")
WRITTEN_BY = ("paper", "written_by", "author")


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

