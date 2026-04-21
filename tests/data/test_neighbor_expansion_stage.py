import torch
import inspect
import pytest

from vgl import Graph
from vgl.dataloading.executor import (
    PlanExecutor,
    _build_stitched_hetero_link_records,
    _build_stitched_homo_link_records,
    _build_stitched_hetero_temporal_record,
    _build_stitched_homo_temporal_record,
    _collect_stitched_hetero_edges,
    _collect_stitched_homo_edges,
    _expand_stitched_hetero_global_node_ids,
    _expand_stitched_homo_global_node_ids,
    _incident_edge_positions,
    _lookup_positions,
    _relabel_stitched_edge_index,
    _relabel_stitched_edge_index_by_type,
    _stitched_hetero_link_seed_global_ids,
    _stitched_hetero_temporal_seed_global_ids,
)
from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.requests import NodeSeedRequest
from vgl.dataloading.records import LinkPredictionRecord, TemporalEventRecord
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


def test_executor_lookup_positions_avoids_tensor_item_in_missing_id_errors(monkeypatch):
    def fail_item(self):
        raise AssertionError("executor lookup should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    with pytest.raises(KeyError, match="missing stitched node id 7"):
        _lookup_positions(torch.tensor([0, 2, 6]), torch.tensor([7]), entity_name="stitched node")

    with pytest.raises(KeyError, match="missing stitched node id 4"):
        _lookup_positions(torch.tensor([0, 2, 6]), torch.tensor([2, 4]), entity_name="stitched node")


def test_executor_lookup_positions_avoids_tensor_int_in_missing_id_errors(monkeypatch):
    def fail_int(self):
        raise AssertionError("executor lookup should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    with pytest.raises(KeyError, match="missing stitched node id 7"):
        _lookup_positions(torch.tensor([0, 2, 6]), torch.tensor([7]), entity_name="stitched node")

    with pytest.raises(KeyError, match="missing stitched node id 4"):
        _lookup_positions(torch.tensor([0, 2, 6]), torch.tensor([2, 4]), entity_name="stitched node")


def test_build_stitched_homo_temporal_record_avoids_tensor_int(monkeypatch):
    graph = Graph.temporal(
        nodes={
            "node": {
                "x": torch.randn(3, 2),
                "n_id": torch.tensor([10, 11, 12]),
            }
        },
        edges={
            ("node", "to", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "ts": torch.tensor([3, 4]),
            }
        },
        time_attr="ts",
    )
    stitched_graph = Graph.temporal(
        nodes={
            "node": {
                "x": torch.randn(4, 2),
                "n_id": torch.tensor([9, 11, 12, 13]),
            }
        },
        edges={
            ("node", "to", "node"): {
                "edge_index": torch.tensor([[1], [2]]),
                "ts": torch.tensor([4]),
            }
        },
        time_attr="ts",
    )

    def fail_int(self):
        raise AssertionError("stitched homo temporal record should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    record = _build_stitched_homo_temporal_record(
        graph,
        TemporalEventRecord(
            graph=graph,
            src_index=torch.tensor(1),
            dst_index=torch.tensor(2),
            timestamp=torch.tensor(7),
            label=torch.tensor(1),
        ),
        stitched_graph,
    )

    assert record.graph is stitched_graph
    assert record.src_index == 1
    assert record.dst_index == 2
    assert record.timestamp == 7
    assert record.label == 1


def test_build_stitched_homo_temporal_record_uses_resolved_metadata_ids():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 2), "n_id": torch.tensor([10, 11, 12])}},
        edges={("node", "to", "node"): {"edge_index": torch.tensor([[0, 1], [1, 2]]), "ts": torch.tensor([3, 4])}},
        time_attr="ts",
    )
    stitched_graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(4, 2), "n_id": torch.tensor([9, 11, 12, 13])}},
        edges={("node", "to", "node"): {"edge_index": torch.tensor([[1], [2]]), "ts": torch.tensor([4])}},
        time_attr="ts",
    )

    record = _build_stitched_homo_temporal_record(
        graph,
        TemporalEventRecord(
            graph=graph,
            src_index=1,
            dst_index=2,
            timestamp=7,
            label=1,
            metadata={"sample_id": "evt-7", "query_id": "q-7"},
        ),
        stitched_graph,
    )

    assert record.sample_id == "evt-7"
    assert record.query_id == "q-7"
    assert record.metadata["sample_id"] == "evt-7"
    assert record.metadata["query_id"] == "q-7"
    assert record.metadata["edge_type"] == ("node", "to", "node")


def test_build_stitched_hetero_temporal_record_avoids_tensor_int(monkeypatch):
    graph = Graph.temporal(
        nodes={
            "author": {
                "x": torch.randn(2, 2),
                "n_id": torch.tensor([10, 11]),
            },
            "paper": {
                "x": torch.randn(3, 2),
                "n_id": torch.tensor([20, 21, 22]),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "ts": torch.tensor([3, 4]),
            }
        },
        time_attr="ts",
    )
    stitched_graph = Graph.temporal(
        nodes={
            "author": {
                "x": torch.randn(2, 2),
                "n_id": torch.tensor([8, 10]),
            },
            "paper": {
                "x": torch.randn(3, 2),
                "n_id": torch.tensor([19, 21, 22]),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[1], [1]]),
                "ts": torch.tensor([4]),
            }
        },
        time_attr="ts",
    )

    def fail_int(self):
        raise AssertionError("stitched hetero temporal record should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    record = _build_stitched_hetero_temporal_record(
        graph,
        TemporalEventRecord(
            graph=graph,
            src_index=torch.tensor(0),
            dst_index=torch.tensor(1),
            timestamp=torch.tensor(9),
            label=torch.tensor(1),
            edge_type=("author", "writes", "paper"),
        ),
        stitched_graph,
    )

    assert record.graph is stitched_graph
    assert record.edge_type == ("author", "writes", "paper")
    assert record.src_index == 1
    assert record.dst_index == 1
    assert record.timestamp == 9
    assert record.label == 1


def test_build_stitched_hetero_temporal_record_uses_resolved_metadata_ids():
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.randn(2, 2), "n_id": torch.tensor([10, 11])},
            "paper": {"x": torch.randn(3, 2), "n_id": torch.tensor([20, 21, 22])},
        },
        edges={("author", "writes", "paper"): {"edge_index": torch.tensor([[0, 1], [1, 2]]), "ts": torch.tensor([3, 4])}},
        time_attr="ts",
    )
    stitched_graph = Graph.temporal(
        nodes={
            "author": {"x": torch.randn(2, 2), "n_id": torch.tensor([8, 10])},
            "paper": {"x": torch.randn(3, 2), "n_id": torch.tensor([19, 21, 22])},
        },
        edges={("author", "writes", "paper"): {"edge_index": torch.tensor([[1], [1]]), "ts": torch.tensor([4])}},
        time_attr="ts",
    )

    record = _build_stitched_hetero_temporal_record(
        graph,
        TemporalEventRecord(
            graph=graph,
            src_index=0,
            dst_index=1,
            timestamp=9,
            label=1,
            metadata={"sample_id": "evt-9", "query_id": "q-9"},
            edge_type=("author", "writes", "paper"),
        ),
        stitched_graph,
    )

    assert record.sample_id == "evt-9"
    assert record.query_id == "q-9"
    assert record.metadata["sample_id"] == "evt-9"
    assert record.metadata["query_id"] == "q-9"
    assert record.metadata["edge_type"] == ("author", "writes", "paper")


def test_build_stitched_homo_link_records_avoids_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 2),
    )
    graph.nodes["node"].data["n_id"] = torch.tensor([10, 11, 12])
    stitched_graph = Graph.homo(
        edge_index=torch.tensor([[1], [2]]),
        x=torch.randn(4, 2),
    )
    stitched_graph.nodes["node"].data["n_id"] = torch.tensor([9, 11, 12, 13])

    def fail_int(self):
        raise AssertionError("stitched homo link records should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    records = _build_stitched_homo_link_records(
        graph,
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(1),
                dst_index=torch.tensor(2),
                label=torch.tensor(1),
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(2),
                dst_index=torch.tensor(1),
                label=torch.tensor(0),
            ),
        ],
        stitched_graph,
    )

    assert [record.graph for record in records] == [stitched_graph, stitched_graph]
    assert [record.src_index for record in records] == [1, 2]
    assert [record.dst_index for record in records] == [2, 1]
    assert [record.label for record in records] == [1, 0]


def test_build_stitched_homo_link_records_use_resolved_metadata_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 2),
    )
    graph.nodes["node"].data["n_id"] = torch.tensor([10, 11, 12])
    stitched_graph = Graph.homo(
        edge_index=torch.tensor([[1], [2]]),
        x=torch.randn(4, 2),
    )
    stitched_graph.nodes["node"].data["n_id"] = torch.tensor([9, 11, 12, 13])

    records = _build_stitched_homo_link_records(
        graph,
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=1,
                dst_index=2,
                label=1,
                metadata={"sample_id": "pos-1", "query_id": "query-a"},
            ),
        ],
        stitched_graph,
    )

    assert records[0].sample_id == "pos-1"
    assert records[0].query_id == "query-a"
    assert records[0].metadata["sample_id"] == "pos-1"
    assert records[0].metadata["query_id"] == "query-a"


def test_build_stitched_homo_link_records_normalize_record_level_routing_metadata():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 2),
    )
    graph.nodes["node"].data["n_id"] = torch.tensor([10, 11, 12])
    stitched_graph = Graph.homo(
        edge_index=torch.tensor([[1], [2]]),
        x=torch.randn(4, 2),
    )
    stitched_graph.nodes["node"].data["n_id"] = torch.tensor([9, 11, 12, 13])

    records = _build_stitched_homo_link_records(
        graph,
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=1,
                dst_index=2,
                label=0,
                sample_id="neg-1",
                query_id="query-c",
                exclude_seed_edge=True,
                hard_negative_dst=[2],
                candidate_dst=[0, 2],
                filter_ranking=True,
            ),
        ],
        stitched_graph,
    )

    assert records[0].metadata["sample_id"] == "neg-1"
    assert records[0].metadata["query_id"] == "query-c"
    assert records[0].metadata["exclude_seed_edges"] is True
    assert records[0].metadata["hard_negative_dst"] == [2]
    assert records[0].metadata["candidate_dst"] == [0, 2]
    assert records[0].metadata["filter_ranking"] is True


def test_build_stitched_hetero_link_records_avoids_tensor_int(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "author": {
                "x": torch.randn(2, 2),
                "n_id": torch.tensor([10, 11]),
            },
            "paper": {
                "x": torch.randn(3, 2),
                "n_id": torch.tensor([20, 21, 22]),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            }
        },
    )
    stitched_graph = Graph.hetero(
        nodes={
            "author": {
                "x": torch.randn(2, 2),
                "n_id": torch.tensor([8, 10]),
            },
            "paper": {
                "x": torch.randn(4, 2),
                "n_id": torch.tensor([19, 20, 21, 22]),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[1, 1], [2, 3]])
            }
        },
    )

    def fail_int(self):
        raise AssertionError("stitched hetero link records should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    records = _build_stitched_hetero_link_records(
        graph,
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(0),
                dst_index=torch.tensor(1),
                label=torch.tensor(1),
                edge_type=("author", "writes", "paper"),
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(0),
                dst_index=torch.tensor(2),
                label=torch.tensor(0),
                edge_type=("author", "writes", "paper"),
            ),
        ],
        stitched_graph,
    )

    assert [record.graph for record in records] == [stitched_graph, stitched_graph]
    assert [record.edge_type for record in records] == [
        ("author", "writes", "paper"),
        ("author", "writes", "paper"),
    ]
    assert [record.src_index for record in records] == [1, 1]
    assert [record.dst_index for record in records] == [2, 3]
    assert [record.label for record in records] == [1, 0]


def test_build_stitched_hetero_link_records_use_resolved_metadata_ids():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 2), "n_id": torch.tensor([10, 11])},
            "paper": {"x": torch.randn(3, 2), "n_id": torch.tensor([20, 21, 22])},
        },
        edges={("author", "writes", "paper"): {"edge_index": torch.tensor([[0, 1], [1, 2]])}},
    )
    stitched_graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 2), "n_id": torch.tensor([8, 10])},
            "paper": {"x": torch.randn(4, 2), "n_id": torch.tensor([19, 20, 21, 22])},
        },
        edges={("author", "writes", "paper"): {"edge_index": torch.tensor([[1, 1], [2, 3]])}},
    )

    records = _build_stitched_hetero_link_records(
        graph,
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                metadata={"sample_id": "edge-1", "query_id": "query-b"},
                edge_type=("author", "writes", "paper"),
            ),
        ],
        stitched_graph,
    )

    assert records[0].sample_id == "edge-1"
    assert records[0].query_id == "query-b"
    assert records[0].metadata["sample_id"] == "edge-1"
    assert records[0].metadata["query_id"] == "query-b"
    assert records[0].metadata["edge_type"] == ("author", "writes", "paper")


def test_build_stitched_hetero_link_records_normalize_record_level_routing_metadata():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 2), "n_id": torch.tensor([10, 11])},
            "paper": {"x": torch.randn(3, 2), "n_id": torch.tensor([20, 21, 22])},
        },
        edges={("author", "writes", "paper"): {"edge_index": torch.tensor([[0, 1], [1, 2]])}},
    )
    stitched_graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 2), "n_id": torch.tensor([8, 10])},
            "paper": {"x": torch.randn(4, 2), "n_id": torch.tensor([19, 20, 21, 22])},
        },
        edges={("author", "writes", "paper"): {"edge_index": torch.tensor([[1, 1], [2, 3]])}},
    )

    records = _build_stitched_hetero_link_records(
        graph,
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=0,
                sample_id="edge-neg",
                query_id="query-neg",
                exclude_seed_edge=True,
                hard_negative_dst=[2],
                candidate_dst=[1, 2],
                filter_ranking=True,
                edge_type=("author", "writes", "paper"),
            ),
        ],
        stitched_graph,
    )

    assert records[0].metadata["sample_id"] == "edge-neg"
    assert records[0].metadata["query_id"] == "query-neg"
    assert records[0].metadata["edge_type"] == ("author", "writes", "paper")
    assert records[0].metadata["exclude_seed_edges"] is True
    assert records[0].metadata["hard_negative_dst"] == [2]
    assert records[0].metadata["candidate_dst"] == [1, 2]
    assert records[0].metadata["filter_ranking"] is True


def test_stitched_hetero_link_seed_global_ids_avoids_tensor_int(monkeypatch):
    graph = Graph.hetero(
        nodes={
            "author": {
                "x": torch.randn(3, 2),
                "n_id": torch.tensor([10, 11, 12]),
            },
            "paper": {
                "x": torch.randn(4, 2),
                "n_id": torch.tensor([20, 21, 22, 23]),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [2, 3]])
            }
        },
    )

    def fail_int(self):
        raise AssertionError("stitched hetero link seed global ids should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    seed_global_ids_by_type = _stitched_hetero_link_seed_global_ids(
        graph,
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(1),
                dst_index=torch.tensor(3),
                label=torch.tensor(1),
                edge_type=("author", "writes", "paper"),
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=torch.tensor(0),
                dst_index=torch.tensor(2),
                label=torch.tensor(0),
                edge_type=("author", "writes", "paper"),
            ),
        ],
    )

    assert torch.equal(seed_global_ids_by_type["author"], torch.tensor([10, 11]))
    assert torch.equal(seed_global_ids_by_type["paper"], torch.tensor([22, 23]))


def test_stitched_hetero_temporal_seed_global_ids_avoids_tensor_int(monkeypatch):
    graph = Graph.temporal(
        nodes={
            "author": {
                "x": torch.randn(2, 2),
                "n_id": torch.tensor([10, 11]),
            },
            "paper": {
                "x": torch.randn(3, 2),
                "n_id": torch.tensor([20, 21, 22]),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "ts": torch.tensor([3, 4]),
            }
        },
        time_attr="ts",
    )

    def fail_int(self):
        raise AssertionError("stitched hetero temporal seed global ids should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    seed_global_ids_by_type = _stitched_hetero_temporal_seed_global_ids(
        graph,
        TemporalEventRecord(
            graph=graph,
            src_index=torch.tensor(1),
            dst_index=torch.tensor(2),
            timestamp=torch.tensor(9),
            label=torch.tensor(1),
            edge_type=("author", "writes", "paper"),
        ),
        edge_type=("author", "writes", "paper"),
    )

    assert torch.equal(seed_global_ids_by_type["author"], torch.tensor([11]))
    assert torch.equal(seed_global_ids_by_type["paper"], torch.tensor([22]))


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


def test_stitched_homo_neighbor_expansion_avoids_torch_unique(monkeypatch, tmp_path):
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

    def fail_unique(*args, **kwargs):
        raise AssertionError("stitched homo expansion should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

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


def test_stitched_hetero_neighbor_expansion_avoids_torch_unique(monkeypatch, tmp_path):
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

    def fail_unique(*args, **kwargs):
        raise AssertionError("stitched hetero expansion should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

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


def test_incident_edge_positions_avoid_torch_unique(monkeypatch):
    def fail_unique(*args, **kwargs):
        raise AssertionError("incident edge position lookup should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    positions = _incident_edge_positions(
        torch.tensor([[0, 1, 3], [1, 2, 1]]),
        torch.tensor([1]),
    )

    assert torch.equal(positions, torch.tensor([0, 1, 2]))


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
