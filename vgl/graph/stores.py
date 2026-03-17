from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class NodeStore:
    type_name: str
    data: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(slots=True)
class EdgeStore:
    type_name: tuple[str, str, str]
    data: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
