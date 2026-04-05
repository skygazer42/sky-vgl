from collections.abc import Mapping
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from vgl._memory import pin_tensor


def _transfer_data(data: Mapping, *, device=None, dtype=None, non_blocking: bool = False) -> dict:
    transferred = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            transferred[key] = value
            continue
        can_cast = dtype is not None and (value.is_floating_point() or value.is_complex())
        if can_cast:
            transferred[key] = value.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
        else:
            transferred[key] = value.to(device=device, non_blocking=non_blocking)
    return transferred


def _pin_data(data: Mapping) -> dict:
    return {
        key: pin_tensor(value) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


class LazyFeatureMap(Mapping):
    def __init__(
        self,
        values: dict[str, object] | None = None,
        loaders: dict[str, Callable[[], object]] | None = None,
    ):
        self._values = dict(values or {})
        self._loaders = dict(loaders or {})

    def _resolve(self, key: str):
        if key in self._values:
            return self._values[key]
        try:
            loader = self._loaders.pop(key)
        except KeyError as exc:
            raise KeyError(key) from exc
        value = loader()
        self._values[key] = value
        return value

    def __getitem__(self, key: str):
        return self._resolve(key)

    def __iter__(self):
        return iter(dict.fromkeys((*self._values.keys(), *self._loaders.keys())))

    def __len__(self) -> int:
        return len(set(self._values) | set(self._loaders))

    def items(self):
        for key in self:
            yield key, self[key]

    def keys(self):
        return tuple(self)

    def values(self):
        for _, value in self.items():
            yield value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


@dataclass(slots=True)
class NodeStore:
    type_name: str
    data: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_feature_store(cls, type_name, feature_names, feature_store):
        loaders = {}
        for feature_name in feature_names:
            key = ("node", type_name, feature_name)
            count = feature_store.shape(key)[0]
            loaders[feature_name] = lambda key=key, count=count, feature_store=feature_store: feature_store.fetch(
                key,
                torch.arange(count, dtype=torch.long),
            ).values
        return cls(type_name, LazyFeatureMap(loaders=loaders))

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            data = object.__getattribute__(self, "data")
            return data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        except AttributeError as exc:
            raise AttributeError(name) from exc

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return NodeStore(
            type_name=self.type_name,
            data=_transfer_data(
                self.data, device=device, dtype=dtype, non_blocking=non_blocking
            ),
        )

    def pin_memory(self):
        return NodeStore(type_name=self.type_name, data=_pin_data(self.data))


@dataclass(slots=True)
class EdgeStore:
    type_name: tuple[str, str, str]
    data: Mapping[str, object] = field(default_factory=dict)
    adjacency_cache: dict[str, object] = field(default_factory=dict)
    query_cache: dict[object, object] = field(default_factory=dict)

    @classmethod
    def from_storage(cls, type_name, feature_names, feature_store, graph_store):
        values = {}
        loaders = {
            "edge_index": lambda type_name=type_name, graph_store=graph_store: graph_store.edge_index(type_name),
        }
        for feature_name in feature_names:
            if feature_name == "edge_index":
                continue
            key = ("edge", type_name, feature_name)
            loaders[feature_name] = (
                lambda key=key, type_name=type_name, feature_store=feature_store, graph_store=graph_store: feature_store.fetch(
                    key,
                    torch.arange(graph_store.edge_count(type_name), dtype=torch.long),
                ).values
            )
        return cls(type_name, LazyFeatureMap(values=values, loaders=loaders))

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            data = object.__getattribute__(self, "data")
            return data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        except AttributeError as exc:
            raise AttributeError(name) from exc

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return EdgeStore(
            type_name=self.type_name,
            data=_transfer_data(
                self.data, device=device, dtype=dtype, non_blocking=non_blocking
            ),
        )

    def pin_memory(self):
        return EdgeStore(type_name=self.type_name, data=_pin_data(self.data))
