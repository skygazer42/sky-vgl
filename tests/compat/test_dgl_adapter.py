import sys
import types

import torch

from vgl import Graph


class FakeDGLGraph:
    def __init__(self, edges, num_nodes=None):
        self._src, self._dst = edges
        self._num_nodes = num_nodes if num_nodes is not None else int(torch.max(torch.cat(edges)).item()) + 1
        self.ndata = {}
        self.edata = {}

    def edges(self):
        return self._src, self._dst

    def num_nodes(self):
        return self._num_nodes


def test_graph_round_trips_to_dgl_graph(monkeypatch):
    dgl_module = types.ModuleType("dgl")
    dgl_module.graph = lambda edges, num_nodes=None: FakeDGLGraph(edges, num_nodes=num_nodes)

    monkeypatch.setitem(sys.modules, "dgl", dgl_module)

    source = torch.tensor([0, 1])
    destination = torch.tensor([1, 0])
    dgl_graph = dgl_module.graph((source, destination))
    dgl_graph.ndata["x"] = torch.randn(2, 4)
    dgl_graph.edata["edge_attr"] = torch.randn(2, 3)

    graph = Graph.from_dgl(dgl_graph)
    restored = graph.to_dgl()

    assert restored.num_nodes() == dgl_graph.num_nodes()
    assert torch.equal(restored.ndata["x"], dgl_graph.ndata["x"])
    assert torch.equal(restored.edata["edge_attr"], dgl_graph.edata["edge_attr"])
