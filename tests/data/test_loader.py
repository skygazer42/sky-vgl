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
