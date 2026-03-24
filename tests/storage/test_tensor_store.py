import pytest
import torch

from vgl.storage import TensorStore, TensorSlice


class DummyStore(TensorStore):
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


def test_tensor_store_protocol_fetches_rows():
    store = DummyStore(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    result = store.fetch(torch.tensor([2, 0]))

    assert result.index.tolist() == [2, 0]
    assert torch.equal(result.values, torch.tensor([[5.0, 6.0], [1.0, 2.0]]))
    assert store.shape == (3, 2)
    assert store.dtype is torch.float32


def test_tensor_slice_validates_first_dimension_alignment():
    with pytest.raises(ValueError, match="same length"):
        TensorSlice(index=torch.tensor([0, 1]), values=torch.tensor([[1.0, 2.0]]))
