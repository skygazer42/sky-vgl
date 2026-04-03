import torch

from vgl import Graph
from vgl.dataloading.executor import PlanExecutor
from vgl.dataloading.materialize import _align_tensor_slice
from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.requests import NodeSeedRequest
from vgl.distributed import (
    LocalGraphShard,
    LocalSamplingCoordinator,
    StoreBackedSamplingCoordinator,
    write_partitioned_graph,
)
from vgl.storage import FeatureStore, InMemoryTensorStore
from vgl.storage.base import TensorSlice


EDGE_TYPE = ("paper", "cites", "paper")


def test_executor_fetches_node_and_edge_features_from_feature_store():
    feature_store = FeatureStore(
        {
            ("node", "paper", "x"): InMemoryTensorStore(torch.tensor([[1.0], [2.0], [3.0]])),
            ("edge", EDGE_TYPE, "weight"): InMemoryTensorStore(torch.tensor([0.1, 0.2, 0.3, 0.4])),
        }
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([2, 0]), node_type="paper"),
        stages=(
            PlanStage(
                "fetch_node_features",
                params={
                    "node_type": "paper",
                    "feature_names": ("x",),
                    "index_key": "node_ids",
                    "output_key": "node_features",
                },
            ),
            PlanStage(
                "fetch_edge_features",
                params={
                    "edge_type": EDGE_TYPE,
                    "feature_names": ("weight",),
                    "index_key": "edge_ids",
                    "output_key": "edge_features",
                },
            ),
        ),
    )

    context = PlanExecutor().execute(
        plan,
        feature_store=feature_store,
        state={
            "node_ids": torch.tensor([2, 0]),
            "edge_ids": torch.tensor([3, 1]),
        },
    )

    node_slice = context.state["node_features"]["x"]
    edge_slice = context.state["edge_features"]["weight"]

    assert node_slice.index.tolist() == [2, 0]
    assert torch.equal(node_slice.values, torch.tensor([[3.0], [1.0]]))
    assert edge_slice.index.tolist() == [3, 1]
    assert torch.equal(edge_slice.values, torch.tensor([0.4, 0.2]))


def test_align_tensor_slice_reorders_without_tensor_tolist(monkeypatch):
    tensor_slice = TensorSlice(
        index=torch.tensor([5, 3, 4]),
        values=torch.tensor([[10.0], [20.0], [30.0]]),
    )

    def fail_tolist(self):
        raise AssertionError("feature slice alignment should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    aligned = _align_tensor_slice(torch.tensor([3, 5]), tensor_slice)

    assert torch.equal(aligned, torch.tensor([[20.0], [10.0]]))


def test_executor_fetches_node_and_edge_features_from_sampling_coordinator(tmp_path):
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.arange(8, dtype=torch.float32).view(4, 2)},
            "paper": {"x": torch.arange(10, 18, dtype=torch.float32).view(4, 2)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 1], [1, 0, 3, 2, 2]]),
                "score": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9]),
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
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([3, 0]), node_type="paper"),
        stages=(
            PlanStage(
                "fetch_node_features",
                params={
                    "node_type": "paper",
                    "feature_names": ("x",),
                    "index_key": "node_ids",
                    "output_key": "node_features",
                },
            ),
            PlanStage(
                "fetch_edge_features",
                params={
                    "edge_type": writes,
                    "feature_names": ("weight",),
                    "index_key": "edge_ids",
                    "output_key": "edge_features",
                },
            ),
        ),
    )

    context = PlanExecutor().execute(
        plan,
        feature_store=coordinator,
        state={
            "node_ids": torch.tensor([3, 0]),
            "edge_ids": torch.tensor([3, 0, 2]),
        },
    )

    node_slice = context.state["node_features"]["x"]
    edge_slice = context.state["edge_features"]["weight"]

    assert node_slice.index.tolist() == [3, 0]
    assert torch.equal(node_slice.values, torch.tensor([[16.0, 17.0], [10.0, 11.0]]))
    assert edge_slice.index.tolist() == [3, 0, 2]
    assert torch.equal(edge_slice.values, torch.tensor([4.0, 1.0, 3.0]))


def test_executor_falls_back_to_routed_edge_tensor_slice_without_tensor_tolist(monkeypatch, tmp_path):
    writes = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.arange(8, dtype=torch.float32).view(4, 2)},
            "paper": {"x": torch.arange(10, 18, dtype=torch.float32).view(4, 2)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0]),
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
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([0]), node_type="paper"),
        stages=(
            PlanStage(
                "fetch_edge_features",
                params={
                    "edge_type": writes,
                    "feature_names": ("weight",),
                    "index_key": "edge_ids",
                    "output_key": "edge_features",
                },
            ),
        ),
    )

    def fail_fetch_edge_features(key, edge_ids):
        raise KeyError(key)

    def fail_tolist(self):
        raise AssertionError("routed edge feature fallback should stay on tensors")

    monkeypatch.setattr(coordinator, "fetch_edge_features", fail_fetch_edge_features)
    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    context = PlanExecutor().execute(
        plan,
        feature_store=coordinator,
        state={
            "edge_ids": torch.tensor([4, 3, 0]),
        },
    )

    edge_slice = context.state["edge_features"]["weight"]

    assert torch.equal(edge_slice.index, torch.tensor([4, 3, 0]))
    assert torch.equal(edge_slice.values, torch.tensor([9.0, 4.0, 1.0]))


def test_executor_fetches_node_and_edge_features_from_store_backed_sampling_coordinator(tmp_path):
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.arange(8, dtype=torch.float32).view(4, 2)},
            "paper": {"x": torch.arange(10, 18, dtype=torch.float32).view(4, 2)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 1], [1, 0, 3, 2, 2]]),
                "score": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    coordinator = StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([3, 0]), node_type="paper"),
        stages=(
            PlanStage(
                "fetch_node_features",
                params={
                    "node_type": "paper",
                    "feature_names": ("x",),
                    "index_key": "node_ids",
                    "output_key": "node_features",
                },
            ),
            PlanStage(
                "fetch_edge_features",
                params={
                    "edge_type": writes,
                    "feature_names": ("weight",),
                    "index_key": "edge_ids",
                    "output_key": "edge_features",
                },
            ),
        ),
    )

    context = PlanExecutor().execute(
        plan,
        feature_store=coordinator,
        state={
            "node_ids": torch.tensor([3, 0]),
            "edge_ids": torch.tensor([3, 0, 2]),
        },
    )

    node_slice = context.state["node_features"]["x"]
    edge_slice = context.state["edge_features"]["weight"]

    assert node_slice.index.tolist() == [3, 0]
    assert torch.equal(node_slice.values, torch.tensor([[16.0, 17.0], [10.0, 11.0]]))
    assert edge_slice.index.tolist() == [3, 0, 2]
    assert torch.equal(edge_slice.values, torch.tensor([4.0, 1.0, 3.0]))


def test_executor_fetches_features_from_typed_state_entries():
    feature_store = FeatureStore(
        {
            ("node", "paper", "x"): InMemoryTensorStore(torch.tensor([[1.0], [2.0], [3.0]])),
            ("edge", EDGE_TYPE, "weight"): InMemoryTensorStore(torch.tensor([0.1, 0.2, 0.3, 0.4])),
        }
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([2, 0]), node_type="paper"),
        stages=(
            PlanStage(
                "fetch_node_features",
                params={
                    "node_type": "paper",
                    "feature_names": ("x",),
                    "index_key": "node_ids_by_type",
                    "output_key": "node_features",
                },
            ),
            PlanStage(
                "fetch_edge_features",
                params={
                    "edge_type": EDGE_TYPE,
                    "feature_names": ("weight",),
                    "index_key": "edge_ids_by_type",
                    "output_key": "edge_features",
                },
            ),
        ),
    )

    context = PlanExecutor().execute(
        plan,
        feature_store=feature_store,
        state={
            "node_ids_by_type": {"paper": torch.tensor([2, 0])},
            "edge_ids_by_type": {EDGE_TYPE: torch.tensor([3, 1])},
        },
    )

    node_slice = context.state["node_features"]["x"]
    edge_slice = context.state["edge_features"]["weight"]

    assert node_slice.index.tolist() == [2, 0]
    assert torch.equal(node_slice.values, torch.tensor([[3.0], [1.0]]))
    assert edge_slice.index.tolist() == [3, 1]
    assert torch.equal(edge_slice.values, torch.tensor([0.4, 0.2]))
