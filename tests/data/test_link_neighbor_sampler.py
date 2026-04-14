import torch
import inspect

from vgl import Graph, HeteroBlock
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import LinkPredictionRecord
from vgl.data.sampler import CandidateLinkSampler, HardNegativeLinkSampler, LinkNeighborSampler, UniformNegativeLinkSampler
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


def test_link_neighbor_sampler_homo_frontier_expansion_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])
    record = LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1)

    def fail_tolist(self):
        raise AssertionError("homo frontier expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sampled = sampler._sample_node_ids(graph, [record])

    assert torch.equal(sampled, torch.tensor([0, 1, 2, 3]))


def test_link_neighbor_sampler_homo_next_frontier_avoids_tensor_item(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])

    def fail_item(self):
        raise AssertionError("homo next frontier should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    sampled = sampler._next_frontier(graph, frontier={0, 1}, visited={0, 1}, fanout=-1)

    assert sampled == {2, 3}


def test_link_neighbor_sampler_homo_next_frontier_avoids_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])

    def fail_int(self):
        raise AssertionError("homo next frontier should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    sampled = sampler._next_frontier(graph, frontier={0, 1}, visited={0, 1}, fanout=-1)

    assert sampled == {2, 3}


def test_link_neighbor_sampler_homo_local_subgraph_avoids_dense_bool_masks(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])
    original_zeros = torch.zeros

    def guarded_zeros(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if kwargs.get("dtype") is torch.bool and size == graph.x.size(0) and caller is not None and caller.f_code.co_name == "_subgraph":
            raise AssertionError("homo local subgraph extraction should avoid dense bool node masks")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)

    subgraph, node_mapping = sampler._subgraph(graph, torch.tensor([0, 1, 2, 3]))

    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1, 3], [1, 2, 1]]))
    assert torch.equal(node_mapping, torch.tensor([0, 1, 2, 3]))


def test_link_neighbor_sampler_homo_local_subgraph_avoids_dense_node_mappings(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])
    original_full = torch.full

    def guarded_full(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if size == (graph.x.size(0),) and caller is not None and caller.f_code.co_name == "_subgraph":
            raise AssertionError("homo local subgraph extraction should avoid dense node mappings")
        return original_full(*args, **kwargs)

    monkeypatch.setattr(torch, "full", guarded_full)

    subgraph, node_mapping = sampler._subgraph(graph, torch.tensor([0, 1, 2, 3]))

    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1, 3], [1, 2, 1]]))
    assert torch.equal(node_mapping, torch.tensor([0, 1, 2, 3]))


def test_link_neighbor_sampler_homo_local_record_avoids_tensor_item(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])

    def fail_item(self):
        raise AssertionError("homo local record mapping should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    record = sampler.sample(LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1))

    assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert record.src_index == 0
    assert record.dst_index == 1


def test_link_neighbor_sampler_homo_local_record_avoids_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])

    def fail_int(self):
        raise AssertionError("homo local record mapping should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    record = sampler.sample(
        LinkPredictionRecord(
            graph=graph,
            src_index=torch.tensor(0),
            dst_index=torch.tensor(1),
            label=torch.tensor(1),
        )
    )

    assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2, 3]))
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


def test_uniform_negative_link_sampler_avoids_tensor_int(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        graph = Graph.homo(
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            x=torch.randn(5, 4),
        )
        sampler = UniformNegativeLinkSampler(num_negatives=2)

        def fail_int(self):
            raise AssertionError("UniformNegativeLinkSampler should stay off tensor.__int__")

        monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

        records = sampler.sample(
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(0),
                dst_index=torch.tensor(1),
                label=torch.tensor(1),
            )
        )

        assert len(records) == 3
        assert all(record.graph is graph for record in records)
        assert records[0].label == 1
        assert all(record.src_index == 0 for record in records)


def test_uniform_negative_link_sampler_accepts_tensor_num_negatives_without_tensor_int(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        graph = Graph.homo(
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            x=torch.randn(5, 4),
        )

        def fail_int(self):
            raise AssertionError("UniformNegativeLinkSampler num_negatives should stay off tensor.__int__")

        monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

        records = UniformNegativeLinkSampler(num_negatives=torch.tensor(2)).sample(
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1)
        )

        assert len(records) == 3
        assert all(record.src_index == 0 for record in records)


def test_uniform_negative_link_sampler_candidate_destinations_avoids_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(4, 4),
    )
    sampler = UniformNegativeLinkSampler()

    def fail_int(self):
        raise AssertionError("UniformNegativeLinkSampler candidate destinations should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    candidates = sampler._candidate_destinations(
        LinkPredictionRecord(
            graph=graph,
            src_index=torch.tensor(0),
            dst_index=torch.tensor(1),
            label=torch.tensor(1),
        )
    )

    assert torch.equal(candidates, torch.tensor([0, 3]))


def test_candidate_link_sampler_avoids_tensor_int(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        graph = Graph.homo(
            edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
            x=torch.randn(4, 4),
        )
        sampler = CandidateLinkSampler()

        def fail_int(self):
            raise AssertionError("CandidateLinkSampler should stay off tensor.__int__")

        monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

        records = sampler.sample(
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(0),
                dst_index=torch.tensor(1),
                label=torch.tensor(1),
            )
        )

        assert len(records) == 4
        assert records[0].label == 1
        assert all(record.src_index == 0 for record in records)


def test_hard_negative_link_sampler_avoids_tensor_int(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        graph = Graph.homo(
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            x=torch.randn(5, 4),
        )
        sampler = HardNegativeLinkSampler(num_negatives=2, num_hard_negatives=2)

        def fail_int(self):
            raise AssertionError("HardNegativeLinkSampler should stay off tensor.__int__")

        monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

        records = sampler.sample(
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(0),
                dst_index=torch.tensor(1),
                label=torch.tensor(1),
                hard_negative_dst=torch.tensor([3, 4]),
            )
        )

        assert len(records) == 3
        assert records[0].label == 1
        assert records[0].dst_index == 1
        assert sorted(record.dst_index for record in records[1:]) == [3, 4]


def test_hard_negative_link_sampler_accepts_tensor_num_hard_negatives_without_tensor_int(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        graph = Graph.homo(
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            x=torch.randn(5, 4),
        )

        def fail_int(self):
            raise AssertionError("HardNegativeLinkSampler num_hard_negatives should stay off tensor.__int__")

        monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

        records = HardNegativeLinkSampler(num_negatives=2, num_hard_negatives=torch.tensor(2)).sample(
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                hard_negative_dst=torch.tensor([3, 4]),
            )
        )

        assert len(records) == 3
        assert sorted(record.dst_index for record in records[1:]) == [3, 4]


def test_uniform_negative_link_sampler_seed_is_reproducible_independent_of_global_rng():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(6, 4),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
    )

    torch.manual_seed(0)
    first = UniformNegativeLinkSampler(num_negatives=4, seed=7).sample(record)

    torch.manual_seed(999)
    second = UniformNegativeLinkSampler(num_negatives=4, seed=7).sample(record)

    assert [int(item.dst_index) for item in first[1:]] == [int(item.dst_index) for item in second[1:]]


def test_hard_negative_link_sampler_seed_is_reproducible_for_hard_candidate_selection():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.randn(8, 4),
    )
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        hard_negative_dst=torch.tensor([3, 4, 5, 6]),
    )

    torch.manual_seed(0)
    first = HardNegativeLinkSampler(num_negatives=3, num_hard_negatives=2, seed=11).sample(record)

    torch.manual_seed(1234)
    second = HardNegativeLinkSampler(num_negatives=3, num_hard_negatives=2, seed=11).sample(record)

    assert [(int(item.dst_index), bool(item.metadata.get("hard_negative_sampled", False))) for item in first[1:]] == [
        (int(item.dst_index), bool(item.metadata.get("hard_negative_sampled", False))) for item in second[1:]
    ]


def test_link_neighbor_sampler_accepts_tensor_ctor_scalars_without_tensor_int(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        graph = Graph.homo(
            edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
            x=torch.arange(12, dtype=torch.float32).view(4, 3),
        )

        def fail_int(self):
            raise AssertionError("LinkNeighborSampler constructor scalars should stay off tensor.__int__")

        monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

        record = LinkNeighborSampler(
            num_neighbors=[torch.tensor(-1)],
            seed=torch.tensor(0),
        ).sample(LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1))

        assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2, 3]))
        assert record.src_index == 0
        assert record.dst_index == 1


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


def test_link_neighbor_sampler_hetero_frontier_expansion_avoids_tensor_tolist(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(3, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 2], [1, 1]])
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 1], [0, 2]])
            },
        },
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        edge_type=("author", "writes", "paper"),
    )

    def fail_tolist(self):
        raise AssertionError("hetero frontier expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sampled = sampler._hetero_sample_node_ids(graph, [record])

    assert torch.equal(sampled["author"], torch.tensor([0, 2]))
    assert torch.equal(sampled["paper"], torch.tensor([1]))


def test_link_neighbor_sampler_hetero_frontier_expansion_avoids_torch_unique(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(3, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 2], [1, 1]])
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 1], [0, 2]])
            },
        },
    )
    sampler = LinkNeighborSampler(num_neighbors=[-1])
    record = LinkPredictionRecord(
        graph=graph,
        src_index=0,
        dst_index=1,
        label=1,
        edge_type=("author", "writes", "paper"),
    )

    def fail_unique(*args, **kwargs):
        raise AssertionError("hetero frontier expansion should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    sampled = sampler._hetero_sample_node_ids(graph, [record])

    assert torch.equal(sampled["author"], torch.tensor([0, 2]))
    assert torch.equal(sampled["paper"], torch.tensor([1]))


def test_link_neighbor_sampler_hetero_local_record_avoids_tensor_item(monkeypatch):
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

    def fail_item(self):
        raise AssertionError("hetero local record mapping should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

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
    assert record.src_index == 0
    assert record.dst_index == 0


def test_link_neighbor_sampler_hetero_local_record_avoids_tensor_int(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[0.0], [1.0]])},
            "paper": {"x": torch.tensor([[2.0], [3.0], [4.0]])},
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

    def fail_int(self):
        raise AssertionError("hetero local record mapping should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    record = sampler.sample(
        LinkPredictionRecord(
            graph=graph,
            src_index=torch.tensor(0),
            dst_index=torch.tensor(1),
            label=torch.tensor(1),
            edge_type=("author", "writes", "paper"),
        )
    )

    assert record.edge_type == ("author", "writes", "paper")
    assert torch.equal(record.graph.nodes["author"].n_id, torch.tensor([0]))
    assert torch.equal(record.graph.nodes["paper"].n_id, torch.tensor([1]))
    assert record.src_index == 0
    assert record.dst_index == 0


def test_link_neighbor_sampler_hetero_local_subgraph_avoids_dense_bool_masks(monkeypatch):
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
    original_zeros = torch.zeros

    def guarded_zeros(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if kwargs.get("dtype") is torch.bool and size in {2, 3} and caller is not None and caller.f_code.co_name == "_hetero_subgraph":
            raise AssertionError("hetero local subgraph extraction should avoid dense bool node masks")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)

    subgraph, node_mapping = sampler._hetero_subgraph(
        graph,
        {
            "author": torch.tensor([0]),
            "paper": torch.tensor([1]),
        },
    )

    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(subgraph.edges[("paper", "written_by", "author")].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(node_mapping["author"], torch.tensor([0]))
    assert torch.equal(node_mapping["paper"], torch.tensor([1]))


def test_link_neighbor_sampler_hetero_local_subgraph_avoids_dense_node_mappings(monkeypatch):
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
    original_full = torch.full

    def guarded_full(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if size in {(2,), (3,)} and caller is not None and caller.f_code.co_name == "_hetero_subgraph":
            raise AssertionError("hetero local subgraph extraction should avoid dense node mappings")
        return original_full(*args, **kwargs)

    monkeypatch.setattr(torch, "full", guarded_full)

    subgraph, node_mapping = sampler._hetero_subgraph(
        graph,
        {
            "author": torch.tensor([0]),
            "paper": torch.tensor([1]),
        },
    )

    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(subgraph.edges[("paper", "written_by", "author")].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(node_mapping["author"], torch.tensor([0]))
    assert torch.equal(node_mapping["paper"], torch.tensor([1]))


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



def test_link_neighbor_sampler_stitched_link_sampling_crosses_partition_boundaries_through_coordinator(monkeypatch, tmp_path):
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

    def fail_tolist(self):
        raise AssertionError("stitched homo link materialization should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    batch = next(iter(loader))

    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(batch.graph.edata["e_id"], torch.tensor([0, 1]))
    assert torch.equal(batch.graph.x, torch.tensor([[0.0], [1.0], [2.0]]))
    assert torch.equal(batch.graph.edata["edge_weight"], torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))
    assert torch.equal(batch.labels, torch.tensor([1.0]))


def test_link_neighbor_sampler_stitched_link_materialization_avoids_tensor_item(monkeypatch, tmp_path):
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

    def fail_item(self):
        raise AssertionError("stitched homo link materialization should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    batch = next(iter(loader))

    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))


def test_link_neighbor_sampler_stitched_link_materialization_avoids_tensor_int(monkeypatch, tmp_path):
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
                    src_index=torch.tensor(0),
                    dst_index=torch.tensor(1),
                    label=torch.tensor(1),
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

    def fail_int(self):
        raise AssertionError("stitched homo link materialization should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    batch = next(iter(loader))

    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))


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



def test_link_neighbor_sampler_stitched_hetero_link_sampling_crosses_partition_boundaries_through_coordinator(monkeypatch, tmp_path):
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

    def fail_tolist(self):
        raise AssertionError("stitched hetero link materialization should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

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


def test_link_neighbor_sampler_stitched_hetero_link_materialization_avoids_tensor_item(monkeypatch, tmp_path):
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

    def fail_item(self):
        raise AssertionError("stitched hetero link materialization should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    batch = next(iter(loader))

    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([0]))


def test_link_neighbor_sampler_stitched_hetero_link_materialization_avoids_torch_unique(monkeypatch, tmp_path):
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

    def fail_unique(*args, **kwargs):
        raise AssertionError("stitched hetero link materialization should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    batch = next(iter(loader))

    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([0]))


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


def test_link_neighbor_sampler_stitched_hetero_output_blocks_materialize_relation_local_blocks_through_coordinator(
    monkeypatch,
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

    def fail_tolist(self):
        raise AssertionError("stitched hetero relation-local blocks should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

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
    monkeypatch,
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

    def fail_tolist(self):
        raise AssertionError("stitched hetero multi-relation blocks should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

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
