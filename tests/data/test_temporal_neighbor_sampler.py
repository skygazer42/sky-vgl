import torch
import inspect

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import TemporalEventRecord
from vgl.data.sampler import TemporalNeighborSampler
from vgl.dataloading.sampler import _sorted_membership_mask
from vgl.dataloading.executor import (
    _build_stitched_hetero_temporal_history,
    _build_stitched_homo_temporal_history,
    _expand_stitched_hetero_temporal_node_ids,
    _expand_stitched_homo_temporal_node_ids,
)
from vgl.distributed import LocalGraphShard, LocalSamplingCoordinator, write_partitioned_graph
from vgl.distributed.coordinator import StoreBackedSamplingCoordinator
from vgl.storage import FeatureStore, InMemoryTensorStore


EDGE_TYPE = ("node", "interacts", "node")


def _store_backed_coordinator(shards):
    root = next(iter(shards.values())).root
    return StoreBackedSamplingCoordinator.from_partition_dir(root)


def _graph():
    return Graph.temporal(
        nodes={"node": {"x": torch.randn(4, 4)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )


def test_temporal_neighbor_sampler_extracts_strict_history_subgraph():
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])

    record = sampler.sample(
        TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1)
    )
    edge_type = EDGE_TYPE

    assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(record.graph.edges[edge_type].edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(record.graph.edges[edge_type].timestamp, torch.tensor([1]))
    assert record.src_index == 1
    assert record.dst_index == 2


def test_temporal_neighbor_sampler_homo_frontier_expansion_avoids_tensor_tolist(monkeypatch):
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    edge_index = graph.edges[EDGE_TYPE].edge_index

    def fail_tolist(self):
        raise AssertionError("temporal frontier expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sampled = sampler._sample_node_ids(edge_index, 1, 2)

    assert torch.equal(sampled, torch.tensor([0, 1, 2]))


def test_temporal_neighbor_sampler_homo_history_subgraph_avoids_dense_bool_masks(monkeypatch):
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    original_zeros = torch.zeros

    def guarded_zeros(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if kwargs.get("dtype") is torch.bool and size == graph.nodes["node"].x.size(0) and caller is not None and caller.f_code.co_name == "_subgraph":
            raise AssertionError("temporal homo history subgraph should avoid dense bool node masks")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)

    subgraph, node_mapping = sampler._subgraph(graph, EDGE_TYPE, torch.tensor([0, 1, 2]), torch.tensor([0, 1]))

    assert torch.equal(subgraph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(subgraph.edges[EDGE_TYPE].edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(node_mapping, torch.tensor([0, 1, 2]))


def test_temporal_neighbor_sampler_homo_history_subgraph_avoids_dense_node_mappings(monkeypatch):
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    original_full = torch.full

    def guarded_full(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if size == (graph.nodes["node"].x.size(0),) and caller is not None and caller.f_code.co_name == "_subgraph":
            raise AssertionError("temporal homo history subgraph should avoid dense node mappings")
        return original_full(*args, **kwargs)

    monkeypatch.setattr(torch, "full", guarded_full)

    subgraph, node_mapping = sampler._subgraph(graph, EDGE_TYPE, torch.tensor([0, 1, 2]), torch.tensor([0, 1]))

    assert torch.equal(subgraph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(subgraph.edges[EDGE_TYPE].edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(node_mapping, torch.tensor([0, 1, 2]))


def test_temporal_homo_history_subgraph_sorted_node_ids_avoid_torch_unique(monkeypatch):
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    original_unique = torch.unique

    def guarded_unique(*args, **kwargs):
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_name == "_positions_for_endpoint_values":
            raise AssertionError("sorted unique temporal node ids should not be uniqued again")
        return original_unique(*args, **kwargs)

    monkeypatch.setattr(torch, "unique", guarded_unique)

    subgraph, node_mapping = sampler._subgraph(graph, EDGE_TYPE, torch.tensor([0, 1, 2]), torch.tensor([0, 1]))

    assert torch.equal(subgraph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(subgraph.edges[EDGE_TYPE].edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(node_mapping, torch.tensor([0, 1, 2]))


def test_temporal_neighbor_sampler_history_edge_filtering_avoids_dense_bool_masks(monkeypatch):
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    original_zeros = torch.zeros

    def guarded_zeros(*args, **kwargs):
        caller = inspect.currentframe().f_back
        if kwargs.get("dtype") is torch.bool and caller is not None and caller.f_code.co_name == "_sorted_membership_mask":
            raise AssertionError("temporal history edge filtering should avoid dense bool membership masks")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)

    record = sampler.sample(
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=6, label=1)
    )

    assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(record.graph.edges[EDGE_TYPE].edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))
    assert torch.equal(record.graph.edges[EDGE_TYPE].timestamp, torch.tensor([1, 3, 5]))


def test_sorted_membership_mask_contiguous_ranges_avoid_searchsorted(monkeypatch):
    original_searchsorted = torch.searchsorted

    def guarded_searchsorted(*args, **kwargs):
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_name == "_sorted_membership_mask":
            raise AssertionError("contiguous sorted membership checks should avoid searchsorted")
        return original_searchsorted(*args, **kwargs)

    monkeypatch.setattr(torch, "searchsorted", guarded_searchsorted)

    mask = _sorted_membership_mask(
        torch.tensor([0, 2, 3, 5]),
        torch.tensor([1, 2, 3, 4]),
    )

    assert torch.equal(mask, torch.tensor([False, True, True, False]))


def test_temporal_neighbor_sampler_can_limit_history_by_max_events():
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1], max_events=1)

    record = sampler.sample(
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=6, label=1)
    )
    edge_type = EDGE_TYPE

    assert torch.equal(record.graph.n_id, torch.tensor([0, 2]))
    assert torch.equal(record.graph.edges[edge_type].edge_index, torch.tensor([[1], [0]]))
    assert torch.equal(record.graph.edges[edge_type].timestamp, torch.tensor([5]))


def test_temporal_neighbor_sampler_history_lookup_avoids_tensor_lt(monkeypatch):
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    original_lt = torch.Tensor.__lt__

    def fail_lt(self, other):
        raise AssertionError("history lookup should avoid elementwise timestamp < comparisons")

    monkeypatch.setattr(torch.Tensor, "__lt__", fail_lt)

    edge_ids = sampler._history_edge_ids(graph, EDGE_TYPE, 3)

    assert torch.equal(edge_ids, torch.tensor([0]))


def test_temporal_neighbor_sampler_inclusive_history_lookup_avoids_tensor_le(monkeypatch):
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1], strict_history=False)
    original_le = torch.Tensor.__le__

    def fail_le(self, other):
        raise AssertionError("inclusive history lookup should avoid elementwise timestamp <= comparisons")

    monkeypatch.setattr(torch.Tensor, "__le__", fail_le)

    edge_ids = sampler._history_edge_ids(graph, EDGE_TYPE, 3)

    assert torch.equal(edge_ids, torch.tensor([0, 1]))


def test_temporal_neighbor_sampler_history_lookup_skips_sort_for_monotonic_timestamps(monkeypatch):
    graph = _graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])

    warmed = sampler._history_edge_ids(graph, EDGE_TYPE, 6)
    assert torch.equal(warmed, torch.tensor([0, 1, 2]))

    def fail_sort(*args, **kwargs):
        raise AssertionError("monotonic temporal history lookup should avoid sorting edge ids")

    monkeypatch.setattr(torch, "sort", fail_sort)

    edge_ids = sampler._history_edge_ids(graph, EDGE_TYPE, 6)

    assert torch.equal(edge_ids, torch.tensor([0, 1, 2]))


def test_temporal_neighbor_sampler_history_lookup_preserves_edge_id_order_for_non_monotonic_timestamps():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(4, 4)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([5, 1, 3]),
            }
        },
        time_attr="timestamp",
    )
    sampler = TemporalNeighborSampler(num_neighbors=[-1])

    edge_ids = sampler._history_edge_ids(graph, EDGE_TYPE, 6)

    assert torch.equal(edge_ids, torch.tensor([0, 1, 2]))


def test_temporal_neighbor_sampler_reuses_cached_endpoint_lookups_after_warmup(monkeypatch):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(5, 4)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
                "timestamp": torch.tensor([1, 2, 3, 4]),
            }
        },
        time_attr="timestamp",
    )
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    record = TemporalEventRecord(graph=graph, src_index=2, dst_index=3, timestamp=4, label=1)

    warmed = sampler.sample(record)

    def fail_argsort(*args, **kwargs):
        raise AssertionError("warm temporal neighbor sampling should reuse cached endpoint lookups")

    monkeypatch.setattr(torch, "argsort", fail_argsort)

    sampled = sampler.sample(record)

    assert torch.equal(sampled.graph.n_id, warmed.graph.n_id)
    assert torch.equal(sampled.graph.edges[EDGE_TYPE].edge_index, warmed.graph.edges[EDGE_TYPE].edge_index)
    assert torch.equal(sampled.graph.edges[EDGE_TYPE].timestamp, warmed.graph.edges[EDGE_TYPE].timestamp)


def test_temporal_neighbor_sampler_relation_frontier_expansion_avoids_tensor_tolist(monkeypatch):
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    edge_index = torch.tensor([[0, 2], [1, 1]])

    def fail_tolist(self):
        raise AssertionError("relation frontier expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sampled = sampler._relation_sample_node_ids(
        edge_index,
        0,
        1,
        src_type="author",
        dst_type="paper",
    )

    assert torch.equal(sampled["author"], torch.tensor([0, 2]))
    assert torch.equal(sampled["paper"], torch.tensor([1]))


def test_temporal_neighbor_sampler_relation_history_subgraph_avoids_dense_bool_masks(monkeypatch):
    graph = _hetero_graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    original_zeros = torch.zeros

    def guarded_zeros(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if kwargs.get("dtype") is torch.bool and size in {2, 3} and caller is not None and caller.f_code.co_name == "_relation_subgraph":
            raise AssertionError("temporal relation history subgraph should avoid dense bool node masks")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)

    subgraph, node_mapping = sampler._relation_subgraph(
        graph,
        HETERO_EDGE_TYPE,
        {
            "author": torch.tensor([1]),
            "paper": torch.tensor([0, 2]),
        },
        torch.tensor([0, 1]),
    )

    assert torch.equal(subgraph.nodes["author"].n_id, torch.tensor([1]))
    assert torch.equal(subgraph.nodes["paper"].n_id, torch.tensor([0, 2]))
    assert torch.equal(subgraph.edges[HETERO_EDGE_TYPE].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(node_mapping["author"], torch.tensor([1]))
    assert torch.equal(node_mapping["paper"], torch.tensor([0, 2]))


def test_temporal_neighbor_sampler_relation_history_subgraph_avoids_dense_node_mappings(monkeypatch):
    graph = _hetero_graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    original_full = torch.full

    def guarded_full(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if size in {(2,), (3,)} and caller is not None and caller.f_code.co_name == "_relation_subgraph":
            raise AssertionError("temporal relation history subgraph should avoid dense node mappings")
        return original_full(*args, **kwargs)

    monkeypatch.setattr(torch, "full", guarded_full)

    subgraph, node_mapping = sampler._relation_subgraph(
        graph,
        HETERO_EDGE_TYPE,
        {
            "author": torch.tensor([1]),
            "paper": torch.tensor([0, 2]),
        },
        torch.tensor([0, 1]),
    )

    assert torch.equal(subgraph.nodes["author"].n_id, torch.tensor([1]))
    assert torch.equal(subgraph.nodes["paper"].n_id, torch.tensor([0, 2]))
    assert torch.equal(subgraph.edges[HETERO_EDGE_TYPE].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(node_mapping["author"], torch.tensor([1]))
    assert torch.equal(node_mapping["paper"], torch.tensor([0, 2]))


def test_stitched_homo_temporal_history_and_expansion_avoid_tensor_tolist(monkeypatch, tmp_path):
    graph = _graph()
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    coordinator = LocalSamplingCoordinator(
        {
            0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
            1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
        }
    )
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    record = TemporalEventRecord(
        graph=graph,
        src_index=1,
        dst_index=2,
        timestamp=3,
        label=1,
        edge_type=EDGE_TYPE,
    )

    def fail_tolist(self):
        raise AssertionError("stitched homo temporal expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    edge_ids, edge_index = _build_stitched_homo_temporal_history(
        graph,
        coordinator,
        sampler,
        record,
        edge_type=EDGE_TYPE,
    )
    sampled = _expand_stitched_homo_temporal_node_ids(
        torch.tensor([1, 2]),
        edge_index,
        fanouts=(-1,),
    )

    assert torch.equal(edge_ids, torch.tensor([0]))
    assert torch.equal(sampled, torch.tensor([0, 1, 2]))


def test_stitched_hetero_temporal_history_and_expansion_avoid_tensor_tolist(monkeypatch, tmp_path):
    graph = _hetero_graph()
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    coordinator = LocalSamplingCoordinator(
        {
            0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
            1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
        }
    )
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    record = TemporalEventRecord(
        graph=graph,
        src_index=1,
        dst_index=2,
        timestamp=5,
        label=1,
        edge_type=HETERO_EDGE_TYPE,
    )

    def fail_tolist(self):
        raise AssertionError("stitched hetero temporal expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    edge_ids, edge_index = _build_stitched_hetero_temporal_history(
        graph,
        coordinator,
        sampler,
        record,
        edge_type=HETERO_EDGE_TYPE,
    )
    sampled = _expand_stitched_hetero_temporal_node_ids(
        {
            "author": torch.tensor([1]),
            "paper": torch.tensor([2]),
        },
        edge_index,
        src_type="author",
        dst_type="paper",
        fanouts=(-1,),
    )

    assert torch.equal(edge_ids, torch.tensor([0, 1]))
    assert torch.equal(sampled["author"], torch.tensor([1]))
    assert torch.equal(sampled["paper"], torch.tensor([0, 2]))


def test_loader_routes_temporal_neighbor_sampler_through_plan_execution():
    graph = _graph()
    dataset = ListDataset(
        [
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1),
        ]
    )

    class PlanOnlyTemporalNeighborSampler(TemporalNeighborSampler):
        def sample(self, item):
            raise AssertionError("loader should use build_plan instead of the legacy sample path")

    loader = Loader(dataset=dataset, sampler=PlanOnlyTemporalNeighborSampler(num_neighbors=[-1]), batch_size=1)

    batch = next(iter(loader))
    edge_type = EDGE_TYPE

    assert torch.equal(batch.timestamp, torch.tensor([3]))
    assert torch.equal(batch.graph.edges[edge_type].timestamp, torch.tensor([1]))

def test_temporal_neighbor_sampler_prefetch_option_materializes_features_into_record_graph():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.zeros(4, 2)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
                "edge_weight": torch.zeros(3),
            }
        },
        time_attr="timestamp",
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])),
            ("edge", EDGE_TYPE, "edge_weight"): InMemoryTensorStore(torch.tensor([10.0, 20.0, 30.0])),
        }
    )
    graph.feature_store = feature_store
    sampler = TemporalNeighborSampler(
        num_neighbors=[-1],
        node_feature_names=("x",),
        edge_feature_names=("edge_weight",),
    )

    record = sampler.sample(
        TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1)
    )

    assert torch.equal(record.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(record.graph.x, torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert torch.equal(record.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([10.0]))


def test_temporal_neighbor_sampler_prefetch_option_materializes_features_into_batch_graph():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.zeros(4, 2)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
                "edge_weight": torch.zeros(3),
            }
        },
        time_attr="timestamp",
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])),
            ("edge", EDGE_TYPE, "edge_weight"): InMemoryTensorStore(torch.tensor([10.0, 20.0, 30.0])),
        }
    )
    loader = Loader(
        dataset=ListDataset(
            [
                TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1),
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=feature_store,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.timestamp, torch.tensor([3]))
    assert torch.equal(batch.graph.x, torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([10.0]))

HETERO_EDGE_TYPE = ("author", "writes", "paper")


def _hetero_graph():
    return Graph.temporal(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            HETERO_EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "timestamp": torch.tensor([1, 4, 6]),
            }
        },
        time_attr="timestamp",
    )


def test_temporal_neighbor_sampler_extracts_relation_local_hetero_history_subgraph():
    graph = _hetero_graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])

    record = sampler.sample(
        TemporalEventRecord(
            graph=graph,
            src_index=1,
            dst_index=2,
            timestamp=5,
            label=1,
            edge_type=HETERO_EDGE_TYPE,
        )
    )

    assert record.edge_type == HETERO_EDGE_TYPE
    assert torch.equal(record.graph.nodes["author"].n_id, torch.tensor([1]))
    assert torch.equal(record.graph.nodes["paper"].n_id, torch.tensor([0, 2]))
    assert torch.equal(record.graph.edges[HETERO_EDGE_TYPE].edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(record.graph.edges[HETERO_EDGE_TYPE].timestamp, torch.tensor([4]))
    assert record.src_index == 0
    assert record.dst_index == 1


def test_temporal_neighbor_sampler_reuses_cached_relation_endpoint_lookups_after_warmup(monkeypatch):
    graph = _hetero_graph()
    sampler = TemporalNeighborSampler(num_neighbors=[-1])
    record = TemporalEventRecord(
        graph=graph,
        src_index=1,
        dst_index=2,
        timestamp=5,
        label=1,
        edge_type=HETERO_EDGE_TYPE,
    )

    warmed = sampler.sample(record)

    def fail_argsort(*args, **kwargs):
        raise AssertionError("warm hetero temporal neighbor sampling should reuse cached endpoint lookups")

    monkeypatch.setattr(torch, "argsort", fail_argsort)

    sampled = sampler.sample(record)

    assert sampled.edge_type == warmed.edge_type
    assert torch.equal(sampled.graph.nodes["author"].n_id, warmed.graph.nodes["author"].n_id)
    assert torch.equal(sampled.graph.nodes["paper"].n_id, warmed.graph.nodes["paper"].n_id)
    assert torch.equal(sampled.graph.edges[HETERO_EDGE_TYPE].edge_index, warmed.graph.edges[HETERO_EDGE_TYPE].edge_index)
    assert torch.equal(sampled.graph.edges[HETERO_EDGE_TYPE].timestamp, warmed.graph.edges[HETERO_EDGE_TYPE].timestamp)


def test_temporal_neighbor_sampler_prefetch_option_materializes_features_into_hetero_record_graph():
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.zeros(2, 2)},
            "paper": {"x": torch.zeros(3, 2)},
        },
        edges={
            HETERO_EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "timestamp": torch.tensor([1, 4, 6]),
                "edge_weight": torch.zeros(3),
            }
        },
        time_attr="timestamp",
    )
    feature_store = FeatureStore(
        {
            ("node", "author", "x"): InMemoryTensorStore(torch.tensor([[10.0, 0.0], [20.0, 0.0]])),
            ("node", "paper", "x"): InMemoryTensorStore(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])),
            ("edge", HETERO_EDGE_TYPE, "edge_weight"): InMemoryTensorStore(torch.tensor([11.0, 22.0, 33.0])),
        }
    )
    graph.feature_store = feature_store
    sampler = TemporalNeighborSampler(
        num_neighbors=[-1],
        node_feature_names={"author": ("x",), "paper": ("x",)},
        edge_feature_names={HETERO_EDGE_TYPE: ("edge_weight",)},
    )

    record = sampler.sample(
        TemporalEventRecord(
            graph=graph,
            src_index=1,
            dst_index=2,
            timestamp=5,
            label=1,
            edge_type=HETERO_EDGE_TYPE,
        )
    )

    assert torch.equal(record.graph.nodes["author"].x, torch.tensor([[20.0, 0.0]]))
    assert torch.equal(record.graph.nodes["paper"].x, torch.tensor([[1.0, 0.0], [3.0, 0.0]]))
    assert torch.equal(record.graph.edges[HETERO_EDGE_TYPE].edge_weight, torch.tensor([22.0]))




def test_temporal_neighbor_sampler_stitched_temporal_sampling_crosses_partition_boundaries_through_coordinator(
    monkeypatch,
    tmp_path,
):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                "timestamp": torch.tensor([1, 3, 5]),
                "edge_weight": torch.tensor([10.0, 20.0, 30.0]),
            }
        },
        time_attr="timestamp",
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
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    timestamp=4,
                    label=1,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    def fail_tolist(self):
        raise AssertionError("stitched homo temporal materialization should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    batch = next(iter(loader))

    assert torch.equal(batch.timestamp, torch.tensor([4]))
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].timestamp, torch.tensor([1, 3]))
    assert torch.equal(batch.graph.x, torch.tensor([[0.0], [1.0], [2.0]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))


def test_temporal_neighbor_sampler_stitched_temporal_sampling_crosses_partition_boundaries_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                "timestamp": torch.tensor([1, 3, 5]),
                "edge_weight": torch.tensor([10.0, 20.0, 30.0]),
            }
        },
        time_attr="timestamp",
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
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=1,
                    timestamp=4,
                    label=1,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names=("x",),
            edge_feature_names=("edge_weight",),
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.timestamp, torch.tensor([4]))
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].timestamp, torch.tensor([1, 3]))
    assert torch.equal(batch.graph.x, torch.tensor([[0.0], [1.0], [2.0]]))
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([1]))



def test_temporal_neighbor_sampler_prefetch_option_keeps_sampled_shard_global_ids_aligned_through_coordinator(tmp_path):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 2], [1, 3]]),
                "timestamp": torch.tensor([1, 3]),
                "edge_weight": torch.tensor([10.0, 20.0]),
            }
        },
        time_attr="timestamp",
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
                TemporalEventRecord(
                    graph=shards[1].graph,
                    src_index=0,
                    dst_index=1,
                    timestamp=4,
                    label=1,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
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
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([20.0]))


def test_temporal_neighbor_sampler_prefetch_option_keeps_sampled_shard_global_ids_aligned_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(4, dtype=torch.float32).view(4, 1)}},
        edges={
            EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 2], [1, 3]]),
                "timestamp": torch.tensor([1, 3]),
                "edge_weight": torch.tensor([10.0, 20.0]),
            }
        },
        time_attr="timestamp",
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
                TemporalEventRecord(
                    graph=shards[1].graph,
                    src_index=0,
                    dst_index=1,
                    timestamp=4,
                    label=1,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
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
    assert torch.equal(batch.graph.edges[EDGE_TYPE].edge_weight, torch.tensor([20.0]))



def test_temporal_neighbor_sampler_stitched_hetero_temporal_sampling_crosses_partition_boundaries_through_coordinator(
    monkeypatch,
    tmp_path,
):
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            HETERO_EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 2, 1], [0, 0, 1]]),
                "timestamp": torch.tensor([1, 3, 6]),
                "edge_weight": torch.tensor([10.0, 20.0, 30.0]),
            }
        },
        time_attr="timestamp",
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
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    timestamp=4,
                    label=1,
                    edge_type=HETERO_EDGE_TYPE,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={HETERO_EDGE_TYPE: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    def fail_tolist(self):
        raise AssertionError("stitched hetero temporal materialization should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    batch = next(iter(loader))

    assert batch.edge_type == HETERO_EDGE_TYPE
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.timestamp, torch.tensor([4]))
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0]))
    assert torch.equal(batch.graph.nodes["author"].x, torch.tensor([[10.0], [30.0]]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[1.0]]))
    assert torch.equal(
        batch.graph.edges[HETERO_EDGE_TYPE].edge_index,
        torch.tensor([[0, 1], [0, 0]]),
    )
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].timestamp, torch.tensor([1, 3]))
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([0]))


def test_temporal_neighbor_sampler_stitched_hetero_temporal_sampling_crosses_partition_boundaries_through_store_backed_coordinator(
    tmp_path,
):
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0]])},
        },
        edges={
            HETERO_EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 2, 1], [0, 0, 1]]),
                "timestamp": torch.tensor([1, 3, 6]),
                "edge_weight": torch.tensor([10.0, 20.0, 30.0]),
            }
        },
        time_attr="timestamp",
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
                TemporalEventRecord(
                    graph=shards[0].graph,
                    src_index=0,
                    dst_index=0,
                    timestamp=4,
                    label=1,
                    edge_type=HETERO_EDGE_TYPE,
                )
            ]
        ),
        sampler=TemporalNeighborSampler(
            num_neighbors=[-1],
            node_feature_names={"author": ("x",), "paper": ("x",)},
            edge_feature_names={HETERO_EDGE_TYPE: ("edge_weight",)},
        ),
        batch_size=1,
        feature_store=coordinator,
    )

    batch = next(iter(loader))

    assert batch.edge_type == HETERO_EDGE_TYPE
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.timestamp, torch.tensor([4]))
    assert torch.equal(batch.graph.nodes["author"].n_id, torch.tensor([0, 2]))
    assert torch.equal(batch.graph.nodes["paper"].n_id, torch.tensor([0]))
    assert torch.equal(batch.graph.nodes["author"].x, torch.tensor([[10.0], [30.0]]))
    assert torch.equal(batch.graph.nodes["paper"].x, torch.tensor([[1.0]]))
    assert torch.equal(
        batch.graph.edges[HETERO_EDGE_TYPE].edge_index,
        torch.tensor([[0, 1], [0, 0]]),
    )
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].e_id, torch.tensor([0, 1]))
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].timestamp, torch.tensor([1, 3]))
    assert torch.equal(batch.graph.edges[HETERO_EDGE_TYPE].edge_weight, torch.tensor([10.0, 20.0]))
    assert torch.equal(batch.src_index, torch.tensor([0]))
    assert torch.equal(batch.dst_index, torch.tensor([0]))
