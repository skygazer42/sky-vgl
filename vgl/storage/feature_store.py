from collections.abc import Mapping
from typing import Any

import torch

from vgl.storage.base import TensorSlice, TensorStore

FeatureKey = tuple[Any, Any, str]


class FeatureStore:
    def __init__(self, stores: Mapping[FeatureKey, TensorStore]):
        self._stores = dict(stores)

    def _store_for(self, key: FeatureKey) -> TensorStore:
        try:
            return self._stores[key]
        except KeyError as exc:
            raise KeyError(f"unknown feature key: {key!r}") from exc

    def fetch(self, key: FeatureKey, index: torch.Tensor) -> TensorSlice:
        return self._store_for(key).fetch(index)

    def shape(self, key: FeatureKey) -> tuple[int, ...]:
        return self._store_for(key).shape
