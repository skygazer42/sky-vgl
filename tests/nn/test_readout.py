import torch

from vgl.nn.readout import global_max_pool, global_mean_pool, global_sum_pool


def test_global_mean_pool_reduces_node_embeddings_per_graph():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    graph_index = torch.tensor([0, 0, 1])

    out = global_mean_pool(x, graph_index)

    assert torch.equal(out, torch.tensor([[2.0, 3.0], [10.0, 20.0]]))


def test_global_sum_and_max_pool_reduce_per_graph():
    x = torch.tensor([[1.0, 2.0], [3.0, 5.0], [10.0, 20.0]])
    graph_index = torch.tensor([0, 0, 1])

    assert torch.equal(global_sum_pool(x, graph_index), torch.tensor([[4.0, 7.0], [10.0, 20.0]]))
    assert torch.equal(global_max_pool(x, graph_index), torch.tensor([[3.0, 5.0], [10.0, 20.0]]))


def test_global_pooling_avoids_tensor_item(monkeypatch):
    x = torch.tensor([[1.0, 2.0], [3.0, 5.0], [10.0, 20.0]])
    graph_index = torch.tensor([0, 0, 1])

    def fail_item(self):
        raise AssertionError("global pooling should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    assert torch.equal(global_sum_pool(x, graph_index), torch.tensor([[4.0, 7.0], [10.0, 20.0]]))
    assert torch.equal(global_mean_pool(x, graph_index), torch.tensor([[2.0, 3.5], [10.0, 20.0]]))
    assert torch.equal(global_max_pool(x, graph_index), torch.tensor([[3.0, 5.0], [10.0, 20.0]]))


def test_global_pooling_avoids_tensor_int(monkeypatch):
    x = torch.tensor([[1.0, 2.0], [3.0, 5.0], [10.0, 20.0]])
    graph_index = torch.tensor([0, 0, 1])

    def fail_int(self):
        raise AssertionError("global pooling should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    assert torch.equal(global_sum_pool(x, graph_index), torch.tensor([[4.0, 7.0], [10.0, 20.0]]))
    assert torch.equal(global_mean_pool(x, graph_index), torch.tensor([[2.0, 3.5], [10.0, 20.0]]))
    assert torch.equal(global_max_pool(x, graph_index), torch.tensor([[3.0, 5.0], [10.0, 20.0]]))
