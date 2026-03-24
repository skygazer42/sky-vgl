import torch

from vgl.storage import InMemoryTensorStore


def test_in_memory_tensor_store_fetches_rows_by_index():
    store = InMemoryTensorStore(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    result = store.fetch(torch.tensor([2, 0]))

    assert result.index.tolist() == [2, 0]
    assert torch.equal(result.values, torch.tensor([[5.0, 6.0], [1.0, 2.0]]))
    assert store.shape == (3, 2)
    assert store.dtype is torch.float32
