import sys
import types

import torch

from vgl import Graph
from vgl.graph import GraphSchema
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore


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


def test_storage_backed_graph_with_sparse_cache_round_trips_to_dgl(monkeypatch):
    dgl_module = types.ModuleType("dgl")
    dgl_module.graph = lambda edges, num_nodes=None: FakeDGLGraph(edges, num_nodes=num_nodes)

    monkeypatch.setitem(sys.modules, "dgl", dgl_module)

    edge_type = ("node", "to", "node")
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(edge_type,),
        node_features={"node": ("x", "n_id")},
        edge_features={edge_type: ("edge_index", "e_id")},
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(torch.tensor([[1.0], [2.0]])),
            ("node", "node", "n_id"): InMemoryTensorStore(torch.tensor([10, 11])),
            ("edge", edge_type, "e_id"): InMemoryTensorStore(torch.tensor([20, 21])),
        }
    )
    graph_store = InMemoryGraphStore(
        edges={edge_type: torch.tensor([[0, 1], [1, 0]])},
        num_nodes={"node": 2},
    )

    graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)
    graph.adjacency(layout="coo")
    restored = Graph.from_dgl(graph.to_dgl())

    assert torch.equal(restored.x, graph.x)
    assert torch.equal(restored.ndata["n_id"], graph.ndata["n_id"])
    assert torch.equal(restored.edata["e_id"], graph.edata["e_id"])
    assert torch.equal(restored.edge_index, graph.edge_index)
