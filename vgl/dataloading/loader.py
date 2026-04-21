from collections import deque
from dataclasses import FrozenInstanceError, fields, is_dataclass

import torch

from vgl._memory import pin_tensor
from vgl.dataloading.executor import PlanExecutor
from vgl.dataloading.materialize import materialize_batch
from vgl.dataloading.plan import SamplingPlan


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _resolve_feature_store(feature_store, graph):
    if feature_store is not None:
        return feature_store
    return getattr(graph, "feature_store", None)


def _resolve_sampled(sampled, executor, feature_store=None):
    if isinstance(sampled, SamplingPlan):
        resolved_feature_store = _resolve_feature_store(feature_store, sampled.graph)
        return executor.execute(sampled, graph=sampled.graph, feature_store=resolved_feature_store)
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
        self.prefetch = _as_python_int(prefetch)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self._worker_loader = None

        if self.prefetch < 0:
            raise ValueError("prefetch must be >= 0")
        if self.prefetch > 0 and self.num_workers > 0:
            raise ValueError("prefetch is only supported with num_workers == 0; use prefetch_factor for worker prefetch")
        if self.prefetch_factor is not None and self.num_workers <= 0:
            raise ValueError("prefetch_factor requires num_workers > 0")
        if self.persistent_workers and self.num_workers <= 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if self.num_workers > 0 and not self._is_map_style_dataset():
            raise TypeError("Loader with workers requires a map-style dataset with __len__ and __getitem__")

    @property
    def prefetch_limit(self) -> int:
        return self.batch_size + self.prefetch

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
            return pin_tensor(value)
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
            field_values = {
                field.name: self._pin_memory_value(getattr(value, field.name))
                for field in fields(value)
            }
            if all(field_values[field.name] is getattr(value, field.name) for field in fields(value)):
                return value
            try:
                for field in fields(value):
                    current = getattr(value, field.name)
                    pinned = field_values[field.name]
                    if pinned is current:
                        continue
                    setattr(value, field.name, pinned)
                return value
            except (AttributeError, FrozenInstanceError, TypeError):
                init_values = {
                    field.name: field_values[field.name]
                    for field in fields(value)
                    if field.init
                }
                rebuilt = type(value)(**init_values)
                for field in fields(value):
                    if field.init:
                        continue
                    object.__setattr__(rebuilt, field.name, field_values[field.name])
                return rebuilt
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
                if not batch_items:
                    continue
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

        fill_pending(self.prefetch_limit)
        while pending:
            batch = []
            for _ in range(min(self.batch_size, len(pending))):
                self._append_sampled(batch, pending.popleft())
            if not batch:
                fill_pending(self.prefetch_limit)
                continue
            yield self._finalize_batch(batch)
            fill_pending(self.prefetch_limit)


DataLoader = Loader
