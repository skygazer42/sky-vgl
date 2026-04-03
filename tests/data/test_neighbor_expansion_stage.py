import torch
import inspect

from vgl import Graph
from vgl.dataloading.executor import (
    PlanExecutor,
    _collect_stitched_hetero_edges,
    _collect_stitched_homo_edges,
    _expand_stitched_hetero_global_node_ids,
    _expand_stitched_homo_global_node_ids,
    _relabel_stitched_edge_index,
    _relabel_stitched_edge_index_by_type,
)
from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.requests import NodeSeedRequest
from vgl.distributed import LocalGraphShard, LocalSamplingCoordinator, write_partitioned_graph


def test_executor_expands_homo_node_seeds_across_neighbor_stages():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([1]), node_type="node"),
        stages=(PlanStage("expand_neighbors", params={"num_neighbors": (-1,)}),),
    )

    context = PlanExecutor().execute(plan, graph=graph)

    assert torch.equal(context.state["node_ids"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(context.state["edge_ids"], torch.tensor([0, 1, 2]))


def test_executor_expands_hetero_node_seeds_by_type():
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4)},
            "author": {"x": torch.randn(2, 4)},
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
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([1]), node_type="paper"),
        stages=(PlanStage("expand_neighbors", params={"num_neighbors": (-1,)}),),
    )

    context = PlanExecutor().execute(plan, graph=graph)

    assert torch.equal(context.state["node_ids_by_type"]["paper"], torch.tensor([1]))
    assert torch.equal(context.state["node_ids_by_type"]["author"], torch.tensor([0]))
    assert torch.equal(
        context.state["edge_ids_by_type"][("author", "writes", "paper")],
        torch.tensor([0]),
    )
    assert torch.equal(
        context.state["edge_ids_by_type"][("paper", "written_by", "author")],
        torch.tensor([0]),
    )


def test_executor_expands_homo_node_seeds_without_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([1]), node_type="node"),
        stages=(PlanStage("expand_neighbors", params={"num_neighbors": (-1,)}),),
    )

    def fail_tolist(self):
        raise AssertionError("homo neighbor expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    context = PlanExecutor().execute(plan, graph=graph)

    assert torch.equal(context.state["node_ids"], torch.tensor([0, 1, 2, 3]))


def test_executor_expands_homo_node_seeds_without_dense_bool_masks(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([1]), node_type="node"),
        stages=(PlanStage("expand_neighbors", params={"num_neighbors": (-1,)}),),
    )
    original_zeros = torch.zeros

    def guarded_zeros(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        caller = inspect.currentframe().f_back
        if kwargs.get("dtype") is torch.bool and size == graph.x.size(0) and caller is not None and caller.f_code.co_name == "_induced_edge_ids":
            raise AssertionError("homo neighbor expansion should avoid dense bool node masks")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)

    context = PlanExecutor().execute(plan, graph=graph)

    assert torch.equal(context.state["node_ids"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(context.state["edge_ids"], torch.tensor([0, 1, 2]))


def test_executor_expands_hetero_node_seeds_without_tensor_tolist(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4)},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 2], [0, 1]])
            },
            ("paper", "cites", "paper"): {
                "edge_index": torch.tensor([[1], [0]])
            },
        },
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([1]), node_type="paper"),
        stages=(PlanStage("expand_neighbors", params={"num_neighbors": (-1,)}),),
    )

    def fail_tolist(self):
        raise AssertionError("hetero neighbor expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    context = PlanExecutor().execute(plan, graph=graph)

    assert torch.equal(context.state["node_ids_by_type"]["paper"], torch.tensor([0, 1]))
    assert torch.equal(context.state["node_ids_by_type"]["author"], torch.tensor([0]))


def test_executor_expands_hetero_node_seeds_without_torch_isin(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4)},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 2], [0, 1]])
            },
            ("paper", "cites", "paper"): {
                "edge_index": torch.tensor([[1], [0]])
            },
        },
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([1]), node_type="paper"),
        stages=(PlanStage("expand_neighbors", params={"num_neighbors": (-1,)}),),
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("hetero neighbor expansion should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    context = PlanExecutor().execute(plan, graph=graph)

    assert torch.equal(context.state["node_ids_by_type"]["paper"], torch.tensor([0, 1]))
    assert torch.equal(context.state["node_ids_by_type"]["author"], torch.tensor([0]))


def test_stitched_homo_neighbor_expansion_avoids_tensor_tolist(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    coordinator = LocalSamplingCoordinator(
        {
            0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
            1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
        }
    )

    def fail_tolist(self):
        raise AssertionError("stitched homo expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sampled = _expand_stitched_homo_global_node_ids(
        coordinator,
        torch.tensor([1]),
        fanouts=(-1,),
        edge_type=graph._default_edge_type(),
    )

    assert torch.equal(sampled, torch.tensor([0, 1, 2, 3]))


def test_stitched_hetero_neighbor_expansion_avoids_tensor_tolist(monkeypatch, tmp_path):
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4)},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 2], [0, 1]])
            },
            ("paper", "cites", "paper"): {
                "edge_index": torch.tensor([[1], [0]])
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    coordinator = LocalSamplingCoordinator(
        {
            0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
            1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
        }
    )

    def fail_tolist(self):
        raise AssertionError("stitched hetero expansion should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    sampled = _expand_stitched_hetero_global_node_ids(
        coordinator,
        {
            "paper": torch.tensor([1]),
            "author": torch.empty(0, dtype=torch.long),
        },
        edge_types=tuple(graph.edges),
        fanouts=(-1,),
    )

    assert torch.equal(sampled["paper"], torch.tensor([0, 1]))
    assert torch.equal(sampled["author"], torch.tensor([0]))


def test_collect_and_relabel_stitched_homo_edges_avoid_tensor_tolist(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    coordinator = LocalSamplingCoordinator(
        {
            0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
            1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
        }
    )

    def fail_tolist(self):
        raise AssertionError("stitched homo edge collection should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    edge_ids, edge_index = _collect_stitched_homo_edges(
        coordinator,
        torch.tensor([0, 1, 2, 3]),
        edge_type=graph._default_edge_type(),
    )
    relabeled = _relabel_stitched_edge_index(torch.tensor([0, 1, 2, 3]), edge_index)

    assert torch.equal(edge_ids, torch.tensor([0, 1, 2]))
    assert torch.equal(relabeled, torch.tensor([[0, 1, 3], [1, 2, 1]]))


def test_collect_and_relabel_stitched_homo_edges_avoid_torch_isin(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    coordinator = LocalSamplingCoordinator(
        {
            0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
            1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
        }
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("stitched homo edge collection should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    edge_ids, edge_index = _collect_stitched_homo_edges(
        coordinator,
        torch.tensor([0, 1, 2, 3]),
        edge_type=graph._default_edge_type(),
    )
    relabeled = _relabel_stitched_edge_index(torch.tensor([0, 1, 2, 3]), edge_index)

    assert torch.equal(edge_ids, torch.tensor([0, 1, 2]))
    assert torch.equal(relabeled, torch.tensor([[0, 1, 3], [1, 2, 1]]))


def test_collect_and_relabel_stitched_hetero_edges_avoid_tensor_tolist(monkeypatch, tmp_path):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4)},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            },
            written_by: {
                "edge_index": torch.tensor([[1, 2], [0, 1]])
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    coordinator = LocalSamplingCoordinator(
        {
            0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
            1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
        }
    )

    def fail_tolist(self):
        raise AssertionError("stitched hetero edge collection should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    edge_ids_by_type, edge_index_by_type = _collect_stitched_hetero_edges(
        coordinator,
        {
            "author": torch.tensor([0]),
            "paper": torch.tensor([1]),
        },
        edge_types=(writes, written_by),
    )
    relabeled = _relabel_stitched_edge_index_by_type(
        {
            "author": torch.tensor([0]),
            "paper": torch.tensor([1]),
        },
        edge_index_by_type,
    )

    assert torch.equal(edge_ids_by_type[writes], torch.tensor([0]))
    assert torch.equal(edge_ids_by_type[written_by], torch.tensor([0]))
    assert torch.equal(relabeled[writes], torch.tensor([[0], [0]]))
    assert torch.equal(relabeled[written_by], torch.tensor([[0], [0]]))


def test_collect_and_relabel_stitched_hetero_edges_avoid_torch_isin(monkeypatch, tmp_path):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4)},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            },
            written_by: {
                "edge_index": torch.tensor([[1, 2], [0, 1]])
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    coordinator = LocalSamplingCoordinator(
        {
            0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
            1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
        }
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("stitched hetero edge collection should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    edge_ids_by_type, edge_index_by_type = _collect_stitched_hetero_edges(
        coordinator,
        {
            "author": torch.tensor([0]),
            "paper": torch.tensor([1]),
        },
        edge_types=(writes, written_by),
    )
    relabeled = _relabel_stitched_edge_index_by_type(
        {
            "author": torch.tensor([0]),
            "paper": torch.tensor([1]),
        },
        edge_index_by_type,
    )

    assert torch.equal(edge_ids_by_type[writes], torch.tensor([0]))
    assert torch.equal(edge_ids_by_type[written_by], torch.tensor([0]))
    assert torch.equal(relabeled[writes], torch.tensor([[0], [0]]))
    assert torch.equal(relabeled[written_by], torch.tensor([[0], [0]]))
