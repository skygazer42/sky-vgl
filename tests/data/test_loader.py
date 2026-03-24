import pytest
import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sampler import FullGraphSampler
from vgl.dataloading.executor import MaterializationContext
from vgl.dataloading.plan import SamplingPlan
from vgl.dataloading.requests import GraphSeedRequest
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

    assert batch.graphs[0].x.is_pinned()


class PlanBackedGraphSampler:
    def build_plan(self, item):
        return SamplingPlan(
            request=GraphSeedRequest(graph_ids=torch.tensor([0]), metadata={"label": 1}),
            graph=item,
        )


class RecordingPlanExecutor:
    def __init__(self):
        self.feature_sources = []

    def execute(self, plan, *, graph=None, feature_store=None, state=None):
        self.feature_sources.append(feature_store)
        return MaterializationContext(request=plan.request, graph=graph, feature_store=feature_store)


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
