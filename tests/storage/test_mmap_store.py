import torch

from vgl.storage import MmapTensorStore


def test_mmap_tensor_store_round_trips_tensor_file(tmp_path):
    path = tmp_path / "tensor.pt"
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    MmapTensorStore.save(path, tensor)
    store = MmapTensorStore(path)
    result = store.fetch(torch.tensor([1, 2]))

    assert result.index.tolist() == [1, 2]
    assert torch.equal(result.values, torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    assert store.shape == (3, 2)
    assert store.dtype is torch.float32
