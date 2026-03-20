from dataclasses import FrozenInstanceError, fields, is_dataclass

import torch

from vgl.dataloading.records import LinkPredictionRecord, SampleRecord, TemporalEventRecord
from vgl.graph.batch import GraphBatch, LinkPredictionBatch, NodeBatch, TemporalEventBatch


class _SampledDataset:
    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.sampler.sample(self.dataset[index])


def _identity_collate(batch):
    return batch


class Loader:
    def __init__(
        self,
        dataset,
        sampler,
        batch_size,
        label_source=None,
        label_key=None,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.label_source = label_source
        self.label_key = label_key
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        if self.prefetch_factor is not None and self.num_workers <= 0:
            raise ValueError("prefetch_factor requires num_workers > 0")
        if self.persistent_workers and self.num_workers <= 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if self.num_workers > 0 and not self._is_map_style_dataset():
            raise TypeError("Loader with workers requires a map-style dataset with __len__ and __getitem__")

    def _is_map_style_dataset(self):
        return hasattr(self.dataset, "__len__") and hasattr(self.dataset, "__getitem__")

    def _dataset_iter(self):
        try:
            return iter(self.dataset)
        except TypeError:
            if self._is_map_style_dataset():
                return (self.dataset[index] for index in range(len(self.dataset)))
            raise TypeError("Loader dataset must be iterable or implement __len__ and __getitem__")

    def _build_batch(self, items):
        if items and isinstance(items[0], TemporalEventRecord):
            return TemporalEventBatch.from_records(items)
        if items and isinstance(items[0], LinkPredictionRecord):
            return LinkPredictionBatch.from_records(items)
        if items and isinstance(items[0], SampleRecord) and items[0].subgraph_seed is not None and self.label_source is None:
            return NodeBatch.from_samples(items)
        if items and hasattr(items[0], "graph") and self.label_source is not None and self.label_key is not None:
            return GraphBatch.from_samples(
                items,
                label_key=self.label_key,
                label_source=self.label_source,
            )
        return GraphBatch.from_graphs(items)

    def _pin_memory_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.pin_memory()
        if isinstance(value, list):
            return [self._pin_memory_value(item) for item in value]
        if isinstance(value, tuple):
            pinned_items = tuple(self._pin_memory_value(item) for item in value)
            if all(pinned is item for pinned, item in zip(pinned_items, value)):
                return value
            return pinned_items
        if isinstance(value, dict):
            return {key: self._pin_memory_value(item) for key, item in value.items()}
        if is_dataclass(value):
            for field in fields(value):
                current = getattr(value, field.name)
                pinned = self._pin_memory_value(current)
                if pinned is current:
                    continue
                try:
                    setattr(value, field.name, pinned)
                except (AttributeError, FrozenInstanceError, TypeError):
                    continue
            return value
        return value

    def _finalize_batch(self, items):
        batch = self._build_batch(items)
        if not self.pin_memory:
            return batch
        if hasattr(batch, "pin_memory"):
            return batch.pin_memory()
        return self._pin_memory_value(batch)

    def __iter__(self):
        if self.num_workers > 0:
            worker_loader = torch.utils.data.DataLoader(
                _SampledDataset(self.dataset, self.sampler),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                collate_fn=_identity_collate,
            )
            for sampled_batch in worker_loader:
                batch_items = []
                for sampled in sampled_batch:
                    if isinstance(sampled, (list, tuple)):
                        batch_items.extend(sampled)
                    else:
                        batch_items.append(sampled)
                yield self._finalize_batch(batch_items)
            return

        batch = []
        seed_count = 0
        for item in self._dataset_iter():
            sampled = self.sampler.sample(item)
            if isinstance(sampled, (list, tuple)):
                batch.extend(sampled)
            else:
                batch.append(sampled)
            seed_count += 1
            if seed_count == self.batch_size:
                yield self._finalize_batch(batch)
                batch = []
                seed_count = 0
        if batch:
            yield self._finalize_batch(batch)


DataLoader = Loader
