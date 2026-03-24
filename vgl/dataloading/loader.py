from collections import deque
from dataclasses import FrozenInstanceError, fields, is_dataclass

import torch

from vgl.dataloading.executor import PlanExecutor
from vgl.dataloading.materialize import materialize_batch, materialize_context
from vgl.dataloading.plan import SamplingPlan


def _resolve_feature_store(feature_store, graph):
    if feature_store is not None:
        return feature_store
    return getattr(graph, "feature_store", None)


def _resolve_sampled(sampled, executor, feature_store=None):
    if isinstance(sampled, SamplingPlan):
        resolved_feature_store = _resolve_feature_store(feature_store, sampled.graph)
        context = executor.execute(sampled, graph=sampled.graph, feature_store=resolved_feature_store)
        return materialize_context(context)
    if isinstance(sampled, (list, tuple)):
        resolved = []
        for value in sampled:
            current = _resolve_sampled(value, executor, feature_store)
            if isinstance(current, list):
                resolved.extend(current)
            elif isinstance(current, tuple):
                resolved.extend(list(current))
            else:
                resolved.append(current)
        return resolved if isinstance(sampled, list) else tuple(resolved)
    return sampled


def _sample_item(sampler, item, executor, feature_store=None):
    build_plan = getattr(sampler, "build_plan", None)
    if callable(build_plan):
        sampled = build_plan(item)
    else:
        sampled = sampler.sample(item)
    return _resolve_sampled(sampled, executor, feature_store)


class _SampledDataset:
    def __init__(self, dataset, sampler, executor, feature_store=None):
        self.dataset = dataset
        self.sampler = sampler
        self.executor = executor
        self.feature_store = feature_store

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return _sample_item(self.sampler, self.dataset[index], self.executor, self.feature_store)


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
        executor=None,
        feature_store=None,
        prefetch=0,
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
        self.executor = PlanExecutor() if executor is None else executor
        self.feature_store = feature_store
        self.prefetch = int(prefetch)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self._worker_loader = None

        if self.prefetch < 0:
            raise ValueError("prefetch must be >= 0")
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

    def _sample_item(self, item):
        return _sample_item(self.sampler, item, self.executor, self.feature_store)

    def _build_batch(self, items):
        return materialize_batch(items, label_source=self.label_source, label_key=self.label_key)

    @staticmethod
    def _append_sampled(batch, sampled):
        if isinstance(sampled, (list, tuple)):
            batch.extend(sampled)
        else:
            batch.append(sampled)

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

    def _make_worker_loader(self):
        return torch.utils.data.DataLoader(
            _SampledDataset(self.dataset, self.sampler, self.executor, self.feature_store),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=_identity_collate,
        )

    def _get_worker_loader(self):
        if self.persistent_workers:
            if self._worker_loader is None:
                self._worker_loader = self._make_worker_loader()
            return self._worker_loader
        return self._make_worker_loader()

    def __iter__(self):
        if self.num_workers > 0:
            worker_loader = self._get_worker_loader()
            for sampled_batch in worker_loader:
                batch_items = []
                for sampled in sampled_batch:
                    self._append_sampled(batch_items, sampled)
                yield self._finalize_batch(batch_items)
            return

        dataset_iter = self._dataset_iter()
        pending = deque()

        def fill_pending(limit):
            while len(pending) < limit:
                try:
                    item = next(dataset_iter)
                except StopIteration:
                    break
                pending.append(self._sample_item(item))

        fill_pending(self.batch_size + self.prefetch)
        while pending:
            batch = []
            for _ in range(min(self.batch_size, len(pending))):
                self._append_sampled(batch, pending.popleft())
            yield self._finalize_batch(batch)
            fill_pending(self.batch_size + self.prefetch)


DataLoader = Loader
