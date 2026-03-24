from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class TensorSlice:
    index: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.index.ndim != 1:
            raise ValueError("TensorSlice index must be rank-1")
        if self.values.ndim == 0:
            raise ValueError("TensorSlice values must be at least rank-1")
        if self.values.size(0) != self.index.numel():
            raise ValueError("TensorSlice index and values must have the same length")


class TensorStore(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def dtype(self) -> torch.dtype:
        ...

    def fetch(self, index: torch.Tensor) -> TensorSlice:
        ...
