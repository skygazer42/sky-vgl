from pathlib import Path
from typing import Union

import torch

from vgl.storage.base import TensorSlice

PathLike = Union[str, Path]


class MmapTensorStore:
    def __init__(self, path: PathLike):
        self._path = Path(path)
        self._tensor: torch.Tensor | None = None

    @staticmethod
    def save(path: PathLike, tensor: torch.Tensor) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, destination)

    def _load(self) -> torch.Tensor:
        if self._tensor is None:
            self._tensor = torch.load(self._path, map_location="cpu", weights_only=True)
        return self._tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._load().shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._load().dtype

    def fetch(self, index: torch.Tensor) -> TensorSlice:
        tensor = self._load()
        return TensorSlice(index=index, values=tensor[index])
