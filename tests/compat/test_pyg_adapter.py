import sys
import types

import pytest
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


def test_graph_to_pyg_supports_featureless_homo_graph(monkeypatch):
    data_module = types.ModuleType("torch_geometric.data")
    data_module.Data = FakeData
    package = types.ModuleType("torch_geometric")
    package.data = data_module

    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)

    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0, 1]),
    )

    restored = graph.to_pyg()

    assert torch.equal(restored.edge_index, graph.edge_index)
    assert restored.num_nodes == 2
    assert torch.equal(restored.y, graph.y)
    assert not hasattr(restored, "x")


def test_graph_from_pyg_supports_featureless_data(monkeypatch):
    data_module = types.ModuleType("torch_geometric.data")
    data_module.Data = FakeData
    package = types.ModuleType("torch_geometric")
    package.data = data_module

    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)

    data = FakeData(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0, 1]),
    )

    graph = Graph.from_pyg(data)

    assert torch.equal(graph.edge_index, data.edge_index)
    assert torch.equal(graph.y, data.y)
    assert "x" not in graph.nodes["node"].data


def test_graph_from_pyg_preserves_num_nodes_for_featureless_data(monkeypatch):
    data_module = types.ModuleType("torch_geometric.data")
    data_module.Data = FakeData
    package = types.ModuleType("torch_geometric")
    package.data = data_module

    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)

    data = FakeData(
        edge_index=torch.tensor([[0], [1]]),
        num_nodes=4,
    )

    graph = Graph.from_pyg(data)

    assert graph._node_count("node") == 4
    assert torch.equal(graph.n_id, torch.tensor([0, 1, 2, 3]))


def test_graph_from_pyg_rejects_inconsistent_num_nodes_with_public_ids(monkeypatch):
    data_module = types.ModuleType("torch_geometric.data")
    data_module.Data = FakeData
    package = types.ModuleType("torch_geometric")
    package.data = data_module

    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)

    data = FakeData(
        edge_index=torch.tensor([[0], [1]]),
        n_id=torch.tensor([10, 11]),
        num_nodes=4,
    )

    with pytest.raises(ValueError, match="must align with num_nodes"):
        Graph.from_pyg(data)


def test_graph_from_pyg_infers_num_nodes_from_edge_index_for_featureless_data(monkeypatch):
    data_module = types.ModuleType("torch_geometric.data")
    data_module.Data = FakeData
    package = types.ModuleType("torch_geometric")
    package.data = data_module

    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)

    data = FakeData(
        edge_index=torch.tensor([[0, 2], [1, 0]]),
    )

    graph = Graph.from_pyg(data)

    assert graph._node_count("node") == 3
    assert torch.equal(graph.n_id, torch.tensor([0, 1, 2]))


def test_graph_to_pyg_preserves_num_nodes_for_featureless_isolated_nodes(monkeypatch):
    data_module = types.ModuleType("torch_geometric.data")
    data_module.Data = FakeData
    package = types.ModuleType("torch_geometric")
    package.data = data_module

    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)

    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        n_id=torch.tensor([0, 1, 2, 3]),
    )

    restored = graph.to_pyg()

    assert restored.num_nodes == 4
    assert torch.equal(restored.edge_index, graph.edge_index)
    assert not hasattr(restored, "x")


def test_graph_round_trips_public_ids_to_pyg_data(monkeypatch):
    data_module = types.ModuleType("torch_geometric.data")
    data_module.Data = FakeData
    package = types.ModuleType("torch_geometric")
    package.data = data_module

    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)

    data = FakeData(
        x=torch.randn(2, 4),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        n_id=torch.tensor([10, 11]),
        e_id=torch.tensor([20, 21]),
    )

    graph = Graph.from_pyg(data)
    restored = graph.to_pyg()

    assert torch.equal(graph.n_id, data.n_id)
    assert torch.equal(graph.edata["e_id"], data.e_id)
    assert torch.equal(restored.n_id, data.n_id)
    assert torch.equal(restored.e_id, data.e_id)
