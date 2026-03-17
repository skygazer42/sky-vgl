import sys
import types

import torch

from vgl import Graph


class FakeData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_graph_round_trips_to_pyg_data(monkeypatch):
    data_module = types.ModuleType("torch_geometric.data")
    data_module.Data = FakeData
    package = types.ModuleType("torch_geometric")
    package.data = data_module

    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)

    data = FakeData(
        x=torch.randn(2, 4),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.randn(2, 3),
        y=torch.tensor([0, 1]),
    )

    graph = Graph.from_pyg(data)
    restored = graph.to_pyg()

    assert torch.equal(restored.edge_index, data.edge_index)
    assert torch.equal(restored.edge_attr, data.edge_attr)
    assert torch.equal(restored.x, data.x)
    assert torch.equal(restored.y, data.y)
