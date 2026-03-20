import pytest
import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sampler import FullGraphSampler


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


class IdentitySampler:
    def sample(self, item):
        return item


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
    dataset = SequenceDataset([1, 2, 3, 4])
    loader = Loader(dataset=dataset, sampler=IdentitySampler(), batch_size=2, num_workers=1)
    loader._build_batch = lambda items: items

    batch = next(iter(loader))

    assert batch == [1, 2]


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
