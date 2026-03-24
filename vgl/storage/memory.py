import torch

from vgl.storage.base import TensorSlice


class InMemoryTensorStore:
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def fetch(self, index: torch.Tensor) -> TensorSlice:
        return TensorSlice(index=index, values=self._tensor[index])
