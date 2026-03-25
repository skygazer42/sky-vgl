import sys
import types

import torch

from vgl import Graph
from vgl.graph import GraphSchema
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore


class FakeDGLDataView:
    def __init__(self):
        self.data = {}


class FakeDGLNodeAccessor:
    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, node_type):
        return self._graph._node_frames[node_type]


class FakeDGLEdgeAccessor:
    def __init__(self, graph):
        self._graph = graph

    def __call__(self, etype=None):
        if etype is None:
            if len(self._graph.canonical_etypes) != 1:
                raise TypeError('etype is required for heterogeneous fake graphs')
            etype = self._graph.canonical_etypes[0]
        return self._graph._edges[tuple(etype)]

    def __getitem__(self, edge_type):
        return self._graph._edge_frames[tuple(edge_type)]


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


class FakeDGLHeteroGraph:
    def __init__(self, data_dict, num_nodes_dict=None):
        self._edges = {tuple(edge_type): tuple(edge_pair) for edge_type, edge_pair in data_dict.items()}
        self.canonical_etypes = tuple(self._edges)
        node_types = []
        for src_type, _, dst_type in self.canonical_etypes:
            if src_type not in node_types:
                node_types.append(src_type)
            if dst_type not in node_types:
                node_types.append(dst_type)
        self.ntypes = tuple(node_types)
        inferred = {node_type: 0 for node_type in self.ntypes}
        for edge_type, (src, dst) in self._edges.items():
            src_type, _, dst_type = edge_type
            if src.numel() > 0:
                inferred[src_type] = max(inferred[src_type], int(src.max().item()) + 1)
            if dst.numel() > 0:
                inferred[dst_type] = max(inferred[dst_type], int(dst.max().item()) + 1)
        self._num_nodes = dict(inferred)
        if num_nodes_dict is not None:
            self._num_nodes.update({str(key): int(value) for key, value in num_nodes_dict.items()})
        self._node_frames = {node_type: FakeDGLDataView() for node_type in self.ntypes}
        self._edge_frames = {edge_type: FakeDGLDataView() for edge_type in self.canonical_etypes}
        self.nodes = FakeDGLNodeAccessor(self)
        self.edges = FakeDGLEdgeAccessor(self)

    def num_nodes(self, node_type=None):
        if node_type is None:
            if len(self.ntypes) != 1:
                raise TypeError('node_type is required for heterogeneous fake graphs')
            node_type = self.ntypes[0]
        return self._num_nodes[node_type]


def _install_fake_dgl(monkeypatch):
    dgl_module = types.ModuleType('dgl')
    dgl_module.graph = lambda edges, num_nodes=None: FakeDGLGraph(edges, num_nodes=num_nodes)
    dgl_module.heterograph = lambda data_dict, num_nodes_dict=None: FakeDGLHeteroGraph(data_dict, num_nodes_dict=num_nodes_dict)
    monkeypatch.setitem(sys.modules, 'dgl', dgl_module)
    return dgl_module


def test_graph_round_trips_to_dgl_graph(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    source = torch.tensor([0, 1])
    destination = torch.tensor([1, 0])
    dgl_graph = dgl_module.graph((source, destination))
    dgl_graph.ndata['x'] = torch.randn(2, 4)
    dgl_graph.edata['edge_attr'] = torch.randn(2, 3)

    graph = Graph.from_dgl(dgl_graph)
    restored = graph.to_dgl()

    assert restored.num_nodes() == dgl_graph.num_nodes()
    assert torch.equal(restored.ndata['x'], dgl_graph.ndata['x'])
    assert torch.equal(restored.edata['edge_attr'], dgl_graph.edata['edge_attr'])


def test_hetero_graph_round_trips_to_dgl_heterograph(monkeypatch):
    _install_fake_dgl(monkeypatch)

    writes = ('author', 'writes', 'paper')
    cites = ('paper', 'cites', 'paper')
    graph = Graph.hetero(
        nodes={
            'author': {'x': torch.tensor([[1.0], [2.0]])},
            'paper': {'x': torch.tensor([[3.0], [4.0], [5.0]])},
        },
        edges={
            writes: {
                'edge_index': torch.tensor([[0, 1], [1, 2]]),
                'weight': torch.tensor([0.5, 1.5]),
            },
            cites: {
                'edge_index': torch.tensor([[0, 1], [1, 2]]),
                'score': torch.tensor([2.0, 3.0]),
            },
        },
    )

    dgl_graph = graph.to_dgl()
    restored = Graph.from_dgl(dgl_graph)

    assert isinstance(dgl_graph, FakeDGLHeteroGraph)
    assert restored.schema.node_types == ('author', 'paper')
    assert set(restored.schema.edge_types) == {writes, cites}
    assert torch.equal(restored.nodes['author'].x, graph.nodes['author'].x)
    assert torch.equal(restored.edges[writes].weight, graph.edges[writes].weight)
    assert torch.equal(restored.edges[cites].score, graph.edges[cites].score)


def test_from_dgl_imports_external_single_relation_heterograph(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    follows = ('user', 'follows', 'user')
    dgl_graph = dgl_module.heterograph(
        {follows: (torch.tensor([0, 1]), torch.tensor([1, 2]))},
        num_nodes_dict={'user': 3},
    )
    dgl_graph.nodes['user'].data['x'] = torch.tensor([[1.0], [2.0], [3.0]])
    dgl_graph.edges[follows].data['weight'] = torch.tensor([0.25, 0.75])

    graph = Graph.from_dgl(dgl_graph)

    assert graph.schema.node_types == ('user',)
    assert graph.schema.edge_types == (follows,)
    assert torch.equal(graph.nodes['user'].x, torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.equal(graph.edges[follows].weight, torch.tensor([0.25, 0.75]))


def test_temporal_graph_round_trips_time_attr_through_dgl_adapter(monkeypatch):
    _install_fake_dgl(monkeypatch)

    edge_type = ('node', 'interacts', 'node')
    graph = Graph.temporal(
        nodes={'node': {'x': torch.tensor([[1.0], [2.0], [3.0]])}},
        edges={
            edge_type: {
                'edge_index': torch.tensor([[0, 1], [1, 2]]),
                'timestamp': torch.tensor([3, 5]),
            }
        },
        time_attr='timestamp',
    )

    dgl_graph = graph.to_dgl()
    restored = Graph.from_dgl(dgl_graph)

    assert isinstance(dgl_graph, FakeDGLHeteroGraph)
    assert getattr(dgl_graph, 'vgl_time_attr') == 'timestamp'
    assert restored.schema.time_attr == 'timestamp'
    assert restored.schema.edge_types == (edge_type,)
    assert torch.equal(restored.edges[edge_type].timestamp, torch.tensor([3, 5]))


def test_storage_backed_graph_with_sparse_cache_round_trips_to_dgl(monkeypatch):
    _install_fake_dgl(monkeypatch)

    edge_type = ('node', 'to', 'node')
    schema = GraphSchema(
        node_types=('node',),
        edge_types=(edge_type,),
        node_features={'node': ('x', 'n_id')},
        edge_features={edge_type: ('edge_index', 'e_id')},
    )
    feature_store = FeatureStore(
        {
            ('node', 'node', 'x'): InMemoryTensorStore(torch.tensor([[1.0], [2.0]])),
            ('node', 'node', 'n_id'): InMemoryTensorStore(torch.tensor([10, 11])),
            ('edge', edge_type, 'e_id'): InMemoryTensorStore(torch.tensor([20, 21])),
        }
    )
    graph_store = InMemoryGraphStore(
        edges={edge_type: torch.tensor([[0, 1], [1, 0]])},
        num_nodes={'node': 2},
    )

    graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)
    graph.adjacency(layout='coo')
    restored = Graph.from_dgl(graph.to_dgl())

    assert torch.equal(restored.x, graph.x)
    assert torch.equal(restored.ndata['n_id'], graph.ndata['n_id'])
    assert torch.equal(restored.edata['e_id'], graph.edata['e_id'])
    assert torch.equal(restored.edge_index, graph.edge_index)
