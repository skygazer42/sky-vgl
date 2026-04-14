import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import SampleRecord
from vgl.data.sampler import FullGraphSampler


class CountingSampler(FullGraphSampler):
    def __init__(self):
        self.calls = []

    def sample(self, item):
        self.calls.append(item.sample_id)
        return super().sample(item)


def _sample(sample_id, label):
    return SampleRecord(
        graph=Graph.homo(
            edge_index=torch.tensor([[0], [1]]),
            x=torch.randn(2, 4),
            y=torch.tensor([label]),
        ),
        metadata={"label": label},
        sample_id=sample_id,
    )


def test_loader_prefetch_preserves_batch_order():
    dataset = ListDataset([
        _sample("a", 1),
        _sample("b", 0),
        _sample("c", 1),
        _sample("d", 0),
    ])
    loader = Loader(
        dataset=dataset,
        sampler=FullGraphSampler(),
        batch_size=2,
        label_source="metadata",
        label_key="label",
        prefetch=2,
    )

    labels = [batch.labels.tolist() for batch in loader]

    assert labels == [[1, 0], [1, 0]]


def test_loader_prefetch_is_bounded_before_first_yield():
    dataset = ListDataset([
        _sample("a", 1),
        _sample("b", 0),
        _sample("c", 1),
        _sample("d", 0),
    ])
    sampler = CountingSampler()
    loader = Loader(
        dataset=dataset,
        sampler=sampler,
        batch_size=2,
        label_source="metadata",
        label_key="label",
        prefetch=1,
    )

    iterator = iter(loader)
    first_batch = next(iterator)

    assert torch.equal(first_batch.labels, torch.tensor([1, 0]))
    assert sampler.calls == ["a", "b", "c"]


def test_loader_prefetch_limit_equals_batch_plus_prefetch():
    loader = Loader(
        dataset=ListDataset([_sample("a", 1)]),
        sampler=FullGraphSampler(),
        batch_size=2,
        prefetch=3,
    )

    assert loader.prefetch_limit == 5


def test_loader_zero_prefetch_samples_only_current_batch_before_first_yield():
    dataset = ListDataset([
        _sample("a", 1),
        _sample("b", 0),
        _sample("c", 1),
    ])
    sampler = CountingSampler()
    loader = Loader(
        dataset=dataset,
        sampler=sampler,
        batch_size=2,
        label_source="metadata",
        label_key="label",
        prefetch=0,
    )

    iterator = iter(loader)
    first_batch = next(iterator)

    assert torch.equal(first_batch.labels, torch.tensor([1, 0]))
    assert loader.prefetch_limit == 2
    assert sampler.calls == ["a", "b"]


def test_loader_prefetch_accepts_tensor_scalar_without_tensor_int(monkeypatch):
    dataset = ListDataset([
        _sample("a", 1),
        _sample("b", 0),
        _sample("c", 1),
        _sample("d", 0),
    ])

    def fail_int(self):
        raise AssertionError("Loader prefetch should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    loader = Loader(
        dataset=dataset,
        sampler=FullGraphSampler(),
        batch_size=2,
        label_source="metadata",
        label_key="label",
        prefetch=torch.tensor(1),
    )

    labels = [batch.labels.tolist() for batch in loader]

    assert labels == [[1, 0], [1, 0]]
