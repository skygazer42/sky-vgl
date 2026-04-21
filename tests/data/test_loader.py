from dataclasses import dataclass

import pytest
import torch

from tests.pinning import assert_tensor_pin_state
from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
import vgl.dataloading.materialize as materialize_module
from vgl.data.sampler import FullGraphSampler
from vgl.dataloading.executor import MaterializationContext
from vgl.dataloading.plan import SamplingPlan
from vgl.dataloading.requests import GraphSeedRequest, NodeSeedRequest
from vgl.graph import GraphSchema
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore


class SequenceDataset:
    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class IterableOnlyDataset:
    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)


@dataclass(frozen=True)
class FrozenBatch:
    target: torch.Tensor


def test_loader_returns_graph_batch_for_list_dataset():
    graphs = [
        Graph.homo(
            edge_index=torch.tensor([[0], [1]]),
            x=torch.randn(2, 4),
            y=torch.tensor([0, 1]),
        ),
        Graph.homo(
            edge_index=torch.tensor([[0], [1]]),
            x=torch.randn(2, 4),
            y=torch.tensor([1, 0]),
        ),
    ]

    dataset = ListDataset(graphs)
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert batch.num_graphs == 2


def test_loader_supports_sequence_style_datasets_without_graphs_attribute():
    graphs = [
        Graph.homo(
            edge_index=torch.tensor([[0], [1]]),
            x=torch.randn(2, 4),
            y=torch.tensor([0, 1]),
        ),
        Graph.homo(
            edge_index=torch.tensor([[0], [1]]),
            x=torch.randn(2, 4),
            y=torch.tensor([1, 0]),
        ),
    ]

    dataset = SequenceDataset(graphs)
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert batch.num_graphs == 2


def test_loader_rejects_workers_for_iterable_only_dataset():
    dataset = IterableOnlyDataset([1, 2, 3])

    with pytest.raises(TypeError, match="map-style"):
        Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2, num_workers=1)


def test_loader_rejects_prefetch_factor_without_workers():
    dataset = SequenceDataset([1])

    with pytest.raises(ValueError, match="prefetch_factor"):
        Loader(
            dataset=dataset,
            sampler=FullGraphSampler(),
            batch_size=1,
            prefetch_factor=2,
        )


def test_loader_rejects_prefetch_with_workers():
    dataset = SequenceDataset([1])

    with pytest.raises(ValueError, match="prefetch is only supported with num_workers == 0"):
        Loader(
            dataset=dataset,
            sampler=FullGraphSampler(),
            batch_size=1,
            prefetch=1,
            num_workers=1,
        )


def test_loader_rejects_persistent_workers_without_workers():
    dataset = SequenceDataset([1])

    with pytest.raises(ValueError, match="persistent_workers"):
        Loader(
            dataset=dataset,
            sampler=FullGraphSampler(),
            batch_size=1,
            persistent_workers=True,
        )


def test_loader_supports_workers_for_map_style_dataset():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    dataset = ListDataset([graph, graph])
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2, num_workers=1)

    batch = next(iter(loader))

    assert batch.num_graphs == 2


def test_loader_reuses_internal_worker_loader_with_persistent_workers():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    dataset = ListDataset([graph, graph])
    loader = Loader(
        dataset=dataset,
        sampler=FullGraphSampler(),
        batch_size=2,
        num_workers=1,
        persistent_workers=True,
    )

    first_batch = next(iter(loader))
    first_worker_loader = loader._worker_loader

    second_batch = next(iter(loader))
    second_worker_loader = loader._worker_loader

    assert first_batch.num_graphs == 2
    assert second_batch.num_graphs == 2
    assert first_worker_loader is second_worker_loader


def test_loader_can_pin_built_batch_tensors():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    dataset = ListDataset([graph, graph])
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2, pin_memory=True)

    batch = next(iter(loader))

    assert_tensor_pin_state(batch.graphs[0].x)


def test_loader_rebuilds_frozen_dataclass_batches_when_pinning(monkeypatch):
    loader = Loader(dataset=SequenceDataset([]), sampler=FullGraphSampler(), batch_size=1, pin_memory=True)
    batch = FrozenBatch(target=torch.tensor([1.0], dtype=torch.float32))

    monkeypatch.setattr("vgl.dataloading.loader.pin_tensor", lambda tensor: tensor + 1.0)

    pinned = loader._pin_memory_value(batch)

    assert isinstance(pinned, FrozenBatch)
    assert pinned is not batch
    assert torch.equal(pinned.target, torch.tensor([2.0], dtype=torch.float32))
    assert torch.equal(batch.target, torch.tensor([1.0], dtype=torch.float32))


class PlanBackedGraphSampler:
    def build_plan(self, item):
        return SamplingPlan(
            request=GraphSeedRequest(graph_ids=torch.tensor([0]), metadata={"label": 1}),
            graph=item,
        )


class PlanBackedNodeSampler:
    def build_plan(self, item):
        graph, metadata = item
        return SamplingPlan(
            request=NodeSeedRequest(
                node_ids=torch.tensor([metadata["seed"]]),
                node_type="node",
                metadata=metadata,
            ),
            graph=graph,
        )


class RecordingPlanExecutor:
    def __init__(self):
        self.feature_sources = []

    def execute(self, plan, *, graph=None, feature_store=None, state=None):
        self.feature_sources.append(feature_store)
        return MaterializationContext(request=plan.request, graph=graph, feature_store=feature_store)


class NodePlanExecutor:
    def execute(self, plan, *, graph=None, feature_store=None, state=None):
        return MaterializationContext(
            request=plan.request,
            graph=graph,
            state={"node_ids": torch.tensor([0, 1, 2]).clone()},
            metadata={"sample_id": plan.request.metadata.get("sample_id")},
            feature_store=feature_store,
        )


def test_loader_forwards_feature_store_to_plan_executor():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    executor = RecordingPlanExecutor()
    feature_source = object()
    loader = Loader(
        dataset=ListDataset([graph]),
        sampler=PlanBackedGraphSampler(),
        batch_size=1,
        label_source="metadata",
        label_key="label",
        executor=executor,
        feature_store=feature_source,
    )

    batch = next(iter(loader))

    assert batch.num_graphs == 1
    assert executor.feature_sources == [feature_source]



def _storage_backed_graph():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(("node", "to", "node"),),
        node_features={"node": ("x",)},
        edge_features={("node", "to", "node"): ("edge_index",)},
    )
    feature_store = FeatureStore({("node", "node", "x"): InMemoryTensorStore(torch.randn(2, 4))})
    graph_store = InMemoryGraphStore(
        edges={("node", "to", "node"): torch.tensor([[0], [1]])},
        num_nodes={"node": 2},
    )
    return Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)


def test_loader_forwards_storage_context_feature_store_to_plan_executor():
    graph = _storage_backed_graph()
    executor = RecordingPlanExecutor()
    loader = Loader(
        dataset=ListDataset([graph]),
        sampler=PlanBackedGraphSampler(),
        batch_size=1,
        label_source="metadata",
        label_key="label",
        executor=executor,
    )

    batch = next(iter(loader))

    assert batch.num_graphs == 1
    assert executor.feature_sources == [graph.feature_store]


def test_loader_batches_plan_backed_node_contexts_before_materialization(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.arange(3, dtype=torch.float32).view(3, 1),
        y=torch.tensor([0, 1, 0]),
    )
    dataset = ListDataset(
        [
            (graph, {"seed": 0, "sample_id": "n0"}),
            (graph, {"seed": 1, "sample_id": "n1"}),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=PlanBackedNodeSampler(),
        batch_size=2,
        executor=NodePlanExecutor(),
    )
    real_subgraph = materialize_module._subgraph
    calls = 0

    def counting_subgraph(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_subgraph(*args, **kwargs)

    monkeypatch.setattr(materialize_module, "_subgraph", counting_subgraph)

    batch = next(iter(loader))

    assert calls == 1
    assert torch.equal(batch.graph.n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.seed_index, torch.tensor([0, 1]))
