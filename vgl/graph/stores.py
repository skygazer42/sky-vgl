from dataclasses import dataclass, field

import torch


def _transfer_data(data: dict, *, device=None, dtype=None, non_blocking: bool = False) -> dict:
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


def _pin_data(data: dict) -> dict:
    return {
        key: value.pin_memory() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


@dataclass(slots=True)
class NodeStore:
    type_name: str
    data: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_feature_store(cls, type_name, feature_names, feature_store):
        data = {}
        for feature_name in feature_names:
            key = ("node", type_name, feature_name)
            index = torch.arange(feature_store.shape(key)[0], dtype=torch.long)
            data[feature_name] = feature_store.fetch(key, index).values
        return cls(type_name, data)

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
    data: dict[str, torch.Tensor] = field(default_factory=dict)
    adjacency_cache: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_storage(cls, type_name, feature_names, feature_store, graph_store):
        edge_count = graph_store.edge_count(type_name)
        data = {"edge_index": graph_store.edge_index(type_name)}
        index = torch.arange(edge_count, dtype=torch.long)
        for feature_name in feature_names:
            if feature_name == "edge_index":
                continue
            key = ("edge", type_name, feature_name)
            data[feature_name] = feature_store.fetch(key, index).values
        return cls(type_name, data)

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
