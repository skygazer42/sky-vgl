import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import LinkPredictionRecord
from vgl.data.sampler import LinkNeighborSampler, UniformNegativeLinkSampler


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
