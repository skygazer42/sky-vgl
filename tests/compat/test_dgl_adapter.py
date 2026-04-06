import sys
import types

import pytest
import torch

from vgl import Graph
from vgl.compat.dgl import _node_count
from vgl.graph import Block, GraphSchema, HeteroBlock
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


def test_dgl_node_count_fallback_avoids_tensor_item(monkeypatch):
    class MissingNodeCountGraph:
        def __init__(self):
            self.edges = {
                ("author", "writes", "paper"): types.SimpleNamespace(
                    edge_index=torch.tensor([[0, 1], [1, 2]])
                ),
                ("paper", "cites", "paper"): types.SimpleNamespace(
                    edge_index=torch.tensor([[0, 2], [2, 3]])
                ),
            }

        def _node_count(self, node_type):
            raise ValueError("missing explicit node count")

    def fail_item(self):
        raise AssertionError("DGL node count fallback should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    assert _node_count(MissingNodeCountGraph(), "paper") == 4


def test_dgl_node_count_fallback_avoids_tensor_int(monkeypatch):
    class MissingNodeCountGraph:
        def __init__(self):
            self.edges = {
                ("author", "writes", "paper"): types.SimpleNamespace(
                    edge_index=torch.tensor([[0, 1], [1, 2]])
                ),
                ("paper", "cites", "paper"): types.SimpleNamespace(
                    edge_index=torch.tensor([[0, 2], [2, 3]])
                ),
            }

        def _node_count(self, node_type):
            raise ValueError("missing explicit node count")

    def fail_int(self):
        raise AssertionError("DGL node count fallback should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    assert _node_count(MissingNodeCountGraph(), "paper") == 4


class FakeDGLBlockNodeAccessor:
    def __init__(self, frames):
        self._frames = frames

    def __getitem__(self, node_type):
        return self._frames[node_type]


class FakeDGLBlock:
    is_block = True

    def __init__(self, data_dict, num_src_nodes=None, num_dst_nodes=None):
        self._edges = {tuple(edge_type): tuple(edge_pair) for edge_type, edge_pair in data_dict.items()}
        self.canonical_etypes = tuple(self._edges)
        src_types = []
        dst_types = []
        for src_type, _, dst_type in self.canonical_etypes:
            if src_type not in src_types:
                src_types.append(src_type)
            if dst_type not in dst_types:
                dst_types.append(dst_type)
        self.srctypes = tuple(src_types)
        self.dsttypes = tuple(dst_types)
        self.ntypes = tuple(dict.fromkeys(src_types + dst_types))

        inferred_src = {node_type: 0 for node_type in self.srctypes}
        inferred_dst = {node_type: 0 for node_type in self.dsttypes}
        for edge_type, (src, dst) in self._edges.items():
            src_type, _, dst_type = edge_type
            if src.numel() > 0:
                inferred_src[src_type] = max(inferred_src[src_type], int(src.max().item()) + 1)
            if dst.numel() > 0:
                inferred_dst[dst_type] = max(inferred_dst[dst_type], int(dst.max().item()) + 1)

        self._num_src_nodes = dict(inferred_src)
        self._num_dst_nodes = dict(inferred_dst)
        if num_src_nodes is not None:
            if isinstance(num_src_nodes, dict):
                self._num_src_nodes.update({str(key): int(value) for key, value in num_src_nodes.items()})
            else:
                if len(self.srctypes) != 1:
                    raise TypeError('num_src_nodes dict is required for heterogeneous fake blocks')
                self._num_src_nodes[self.srctypes[0]] = int(num_src_nodes)
        if num_dst_nodes is not None:
            if isinstance(num_dst_nodes, dict):
                self._num_dst_nodes.update({str(key): int(value) for key, value in num_dst_nodes.items()})
            else:
                if len(self.dsttypes) != 1:
                    raise TypeError('num_dst_nodes dict is required for heterogeneous fake blocks')
                self._num_dst_nodes[self.dsttypes[0]] = int(num_dst_nodes)

        self._src_node_frames = {node_type: FakeDGLDataView() for node_type in self.srctypes}
        self._dst_node_frames = {node_type: FakeDGLDataView() for node_type in self.dsttypes}
        self._edge_frames = {edge_type: FakeDGLDataView() for edge_type in self.canonical_etypes}
        self.srcnodes = FakeDGLBlockNodeAccessor(self._src_node_frames)
        self.dstnodes = FakeDGLBlockNodeAccessor(self._dst_node_frames)
        self.edges = FakeDGLEdgeAccessor(self)

    @property
    def srcdata(self):
        if len(self.srctypes) != 1:
            raise TypeError('srcdata is only available for homogeneous fake blocks')
        return self._src_node_frames[self.srctypes[0]].data

    @property
    def dstdata(self):
        if len(self.dsttypes) != 1:
            raise TypeError('dstdata is only available for homogeneous fake blocks')
        return self._dst_node_frames[self.dsttypes[0]].data

    @property
    def edata(self):
        if len(self.canonical_etypes) != 1:
            raise TypeError('edata is only available for single-relation fake blocks')
        return self._edge_frames[self.canonical_etypes[0]].data

    def num_src_nodes(self, node_type=None):
        if node_type is None:
            if len(self.srctypes) != 1:
                raise TypeError('node_type is required for heterogeneous fake blocks')
            node_type = self.srctypes[0]
        return self._num_src_nodes[node_type]

    def num_dst_nodes(self, node_type=None):
        if node_type is None:
            if len(self.dsttypes) != 1:
                raise TypeError('node_type is required for heterogeneous fake blocks')
            node_type = self.dsttypes[0]
        return self._num_dst_nodes[node_type]


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
    dgl_module.NID = 'dgl.NID'
    dgl_module.EID = 'dgl.EID'
    dgl_module.graph = lambda edges, num_nodes=None: FakeDGLGraph(edges, num_nodes=num_nodes)
    dgl_module.heterograph = lambda data_dict, num_nodes_dict=None: FakeDGLHeteroGraph(data_dict, num_nodes_dict=num_nodes_dict)

    def _create_block(data_dict, num_src_nodes=None, num_dst_nodes=None):
        if isinstance(data_dict, dict):
            normalized = data_dict
        else:
            normalized = {('_N', '_E', '_N'): data_dict}
        return FakeDGLBlock(normalized, num_src_nodes=num_src_nodes, num_dst_nodes=num_dst_nodes)

    dgl_module.create_block = _create_block
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


def test_featureless_storage_backed_graph_preserves_num_nodes_on_dgl_export(monkeypatch):
    _install_fake_dgl(monkeypatch)

    edge_type = ('node', 'to', 'node')
    schema = GraphSchema(
        node_types=('node',),
        edge_types=(edge_type,),
        node_features={'node': ()},
        edge_features={edge_type: ('edge_index',)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            edges={edge_type: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={'node': 4},
        ),
    )

    dgl_graph = graph.to_dgl()
    restored = Graph.from_dgl(dgl_graph)

    assert dgl_graph.num_nodes() == 4
    assert torch.equal(restored.ndata['n_id'], torch.tensor([0, 1, 2, 3]))


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



def test_from_dgl_preserves_homo_num_nodes_without_node_features(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    dgl_graph = dgl_module.graph(
        (torch.tensor([0, 1]), torch.tensor([1, 0])),
        num_nodes=4,
    )

    graph = Graph.from_dgl(dgl_graph)
    restored = graph.to_dgl()

    assert torch.equal(graph.ndata['n_id'], torch.tensor([0, 1, 2, 3]))
    assert restored.num_nodes() == 4


def test_from_dgl_avoids_tensor_int_when_num_nodes_returns_tensor(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    dgl_graph = dgl_module.graph(
        (torch.tensor([0, 1]), torch.tensor([1, 0])),
        num_nodes=4,
    )
    dgl_graph.num_nodes = lambda node_type=None: torch.tensor(4)

    def fail_int(self):
        raise AssertionError("Graph.from_dgl should stay off tensor.__int__ for node counts")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    graph = Graph.from_dgl(dgl_graph)

    assert torch.equal(graph.ndata['n_id'], torch.tensor([0, 1, 2, 3]))



def test_from_dgl_preserves_hetero_num_nodes_without_node_features(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    writes = ('author', 'writes', 'paper')
    dgl_graph = dgl_module.heterograph(
        {writes: (torch.tensor([0]), torch.tensor([1]))},
        num_nodes_dict={'author': 3, 'paper': 4},
    )

    graph = Graph.from_dgl(dgl_graph)
    restored = graph.to_dgl()

    assert torch.equal(graph.nodes['author'].data['n_id'], torch.tensor([0, 1, 2]))
    assert torch.equal(graph.nodes['paper'].data['n_id'], torch.tensor([0, 1, 2, 3]))
    assert restored.num_nodes('author') == 3
    assert restored.num_nodes('paper') == 4



def test_from_dgl_normalizes_external_graph_ids(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    dgl_graph = dgl_module.graph(
        (torch.tensor([0, 2]), torch.tensor([1, 0])),
        num_nodes=3,
    )
    dgl_graph.ndata[dgl_module.NID] = torch.tensor([10, 11, 12])
    dgl_graph.edata[dgl_module.EID] = torch.tensor([20, 21])

    graph = Graph.from_dgl(dgl_graph)

    assert torch.equal(graph.ndata['n_id'], torch.tensor([10, 11, 12]))
    assert dgl_module.NID not in graph.ndata
    assert torch.equal(graph.edata['e_id'], torch.tensor([20, 21]))
    assert dgl_module.EID not in graph.edata



def test_graph_from_dgl_rejects_blocks(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    dgl_block = dgl_module.create_block(
        (torch.tensor([0]), torch.tensor([0])),
        num_src_nodes=1,
        num_dst_nodes=1,
    )

    with pytest.raises(ValueError, match='Block.from_dgl'):
        Graph.from_dgl(dgl_block)


def test_homo_block_round_trips_to_dgl_block(monkeypatch):
    _install_fake_dgl(monkeypatch)

    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 1], [1, 2, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        n_id=torch.tensor([10, 11, 12]),
        edge_data={
            'e_id': torch.tensor([20, 21, 22, 23]),
            'weight': torch.tensor([0.5, 0.6, 0.7, 0.8]),
        },
    )

    block = graph.to_block(torch.tensor([1, 2]))

    dgl_block = block.to_dgl()
    restored = Block.from_dgl(dgl_block)

    assert getattr(dgl_block, 'is_block', False)
    assert restored.edge_type == block.edge_type
    assert restored.src_type == block.src_type
    assert restored.dst_type == block.dst_type
    assert torch.equal(restored.src_n_id, block.src_n_id)
    assert torch.equal(restored.dst_n_id, block.dst_n_id)
    assert torch.equal(restored.srcdata['x'], block.srcdata['x'])
    assert torch.equal(restored.dstdata['x'], block.dstdata['x'])
    assert torch.equal(restored.edata['e_id'], block.edata['e_id'])
    assert torch.equal(restored.edata['weight'], block.edata['weight'])
    assert torch.equal(restored.edge_index, block.edge_index)



def test_relation_local_hetero_block_round_trips_to_dgl_block(monkeypatch):
    _install_fake_dgl(monkeypatch)

    writes = ('author', 'writes', 'paper')
    graph = Graph.hetero(
        nodes={
            'author': {'x': torch.tensor([[10.0], [20.0]])},
            'paper': {'x': torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {
                'edge_index': torch.tensor([[0, 1, 1], [1, 0, 2]]),
                'e_id': torch.tensor([30, 31, 32]),
                'weight': torch.tensor([1.0, 2.0, 3.0]),
            }
        },
    )

    block = graph.to_block(torch.tensor([0, 2]), edge_type=writes)

    dgl_block = block.to_dgl()
    restored = Block.from_dgl(dgl_block)

    assert getattr(dgl_block, 'is_block', False)
    assert restored.edge_type == writes
    assert restored.src_type == 'author'
    assert restored.dst_type == 'paper'
    assert torch.equal(restored.src_n_id, block.src_n_id)
    assert torch.equal(restored.dst_n_id, block.dst_n_id)
    assert torch.equal(restored.srcdata['x'], block.srcdata['x'])
    assert torch.equal(restored.dstdata['x'], block.dstdata['x'])
    assert torch.equal(restored.edata['e_id'], block.edata['e_id'])
    assert torch.equal(restored.edata['weight'], block.edata['weight'])
    assert torch.equal(restored.edge_index, block.edge_index)



def test_multi_relation_hetero_block_round_trips_to_dgl_block(monkeypatch):
    _install_fake_dgl(monkeypatch)

    writes = ('author', 'writes', 'paper')
    cites = ('paper', 'cites', 'paper')
    graph = Graph.hetero(
        nodes={
            'author': {'x': torch.tensor([[10.0], [20.0]])},
            'paper': {'x': torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {
                'edge_index': torch.tensor([[0, 1, 1], [1, 0, 2]]),
                'e_id': torch.tensor([30, 31, 32]),
                'weight': torch.tensor([1.0, 2.0, 3.0]),
            },
            cites: {
                'edge_index': torch.tensor([[0, 1, 2], [2, 2, 0]]),
                'e_id': torch.tensor([40, 41, 42]),
                'weight': torch.tensor([5.0, 6.0, 7.0]),
            },
        },
    )

    block = graph.to_hetero_block({'paper': torch.tensor([0, 2])})

    dgl_block = block.to_dgl()
    restored = HeteroBlock.from_dgl(dgl_block)

    assert getattr(dgl_block, 'is_block', False)
    assert restored.edge_types == block.edge_types
    assert restored.src_store_types == block.src_store_types
    assert restored.dst_store_types == block.dst_store_types
    assert torch.equal(restored.src_n_id['author'], block.src_n_id['author'])
    assert torch.equal(restored.src_n_id['paper'], block.src_n_id['paper'])
    assert torch.equal(restored.dst_n_id['paper'], block.dst_n_id['paper'])
    assert torch.equal(restored.srcdata('author')['x'], block.srcdata('author')['x'])
    assert torch.equal(restored.srcdata('paper')['x'], block.srcdata('paper')['x'])
    assert torch.equal(restored.dstdata('paper')['x'], block.dstdata('paper')['x'])
    assert torch.equal(restored.edata(writes)['e_id'], block.edata(writes)['e_id'])
    assert torch.equal(restored.edata(cites)['e_id'], block.edata(cites)['e_id'])
    assert torch.equal(restored.edata(cites)['weight'], block.edata(cites)['weight'])
    assert torch.equal(restored.edge_index(writes), block.edge_index(writes))
    assert torch.equal(restored.edge_index(cites), block.edge_index(cites))


def test_from_dgl_imports_external_single_relation_block(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    follows = ('user', 'follows', 'user')
    dgl_block = dgl_module.create_block(
        {follows: (torch.tensor([2, 0, 1]), torch.tensor([0, 1, 0]))},
        num_src_nodes={'user': 3},
        num_dst_nodes={'user': 2},
    )
    dgl_block.srcnodes['user'].data['n_id'] = torch.tensor([11, 12, 10])
    dgl_block.dstnodes['user'].data['n_id'] = torch.tensor([11, 12])
    dgl_block.srcnodes['user'].data['x'] = torch.tensor([[2.0], [3.0], [1.0]])
    dgl_block.dstnodes['user'].data['x'] = torch.tensor([[2.0], [3.0]])
    dgl_block.edges[follows].data['e_id'] = torch.tensor([40, 41, 42])
    dgl_block.edges[follows].data['weight'] = torch.tensor([0.5, 0.6, 0.7])

    block = Block.from_dgl(dgl_block)

    assert block.edge_type == follows
    assert block.src_type == 'user'
    assert block.dst_type == 'user'
    assert torch.equal(block.src_n_id, torch.tensor([11, 12, 10]))
    assert torch.equal(block.dst_n_id, torch.tensor([11, 12]))
    assert torch.equal(block.srcdata['x'], torch.tensor([[2.0], [3.0], [1.0]]))
    assert torch.equal(block.dstdata['x'], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(block.edata['e_id'], torch.tensor([40, 41, 42]))
    assert torch.equal(block.edata['weight'], torch.tensor([0.5, 0.6, 0.7]))
    assert torch.equal(block.edge_index, torch.tensor([[2, 0, 1], [0, 1, 0]]))


def test_block_from_dgl_avoids_tensor_int_when_num_nodes_returns_tensor(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    follows = ('user', 'follows', 'user')
    dgl_block = dgl_module.create_block(
        {follows: (torch.tensor([2, 0, 1]), torch.tensor([0, 1, 0]))},
        num_src_nodes={'user': 3},
        num_dst_nodes={'user': 2},
    )
    dgl_block.num_src_nodes = lambda node_type=None: torch.tensor(3)
    dgl_block.num_dst_nodes = lambda node_type=None: torch.tensor(2)

    def fail_int(self):
        raise AssertionError("Block.from_dgl should stay off tensor.__int__ for node counts")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    block = Block.from_dgl(dgl_block)

    assert torch.equal(block.src_n_id, torch.tensor([0, 1, 2]))
    assert torch.equal(block.dst_n_id, torch.tensor([0, 1]))



def test_from_dgl_imports_external_multi_relation_block(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    writes = ('author', 'writes', 'paper')
    cites = ('paper', 'cites', 'paper')
    dgl_block = dgl_module.create_block(
        {
            writes: (torch.tensor([0, 0]), torch.tensor([0, 1])),
            cites: (torch.tensor([0, 2, 1]), torch.tensor([1, 1, 0])),
        },
        num_src_nodes={'author': 1, 'paper': 3},
        num_dst_nodes={'paper': 2},
    )
    dgl_block.srcnodes['author'].data['n_id'] = torch.tensor([12])
    dgl_block.srcnodes['author'].data['x'] = torch.tensor([[20.0]])
    dgl_block.srcnodes['paper'].data['n_id'] = torch.tensor([20, 22, 21])
    dgl_block.srcnodes['paper'].data['x'] = torch.tensor([[1.0], [3.0], [2.0]])
    dgl_block.dstnodes['paper'].data['n_id'] = torch.tensor([20, 22])
    dgl_block.dstnodes['paper'].data['x'] = torch.tensor([[1.0], [3.0]])
    dgl_block.edges[writes].data['e_id'] = torch.tensor([31, 32])
    dgl_block.edges[writes].data['weight'] = torch.tensor([2.0, 3.0])
    dgl_block.edges[cites].data['e_id'] = torch.tensor([40, 41, 42])
    dgl_block.edges[cites].data['weight'] = torch.tensor([5.0, 6.0, 7.0])

    block = HeteroBlock.from_dgl(dgl_block)

    assert block.edge_types == (writes, cites)
    assert block.src_store_types == {'author': 'author', 'paper': 'paper__src'}
    assert block.dst_store_types == {'paper': 'paper__dst'}
    assert torch.equal(block.src_n_id['author'], torch.tensor([12]))
    assert torch.equal(block.src_n_id['paper'], torch.tensor([20, 22, 21]))
    assert torch.equal(block.dst_n_id['paper'], torch.tensor([20, 22]))
    assert torch.equal(block.srcdata('author')['x'], torch.tensor([[20.0]]))
    assert torch.equal(block.srcdata('paper')['x'], torch.tensor([[1.0], [3.0], [2.0]]))
    assert torch.equal(block.dstdata('paper')['x'], torch.tensor([[1.0], [3.0]]))
    assert torch.equal(block.edata(writes)['e_id'], torch.tensor([31, 32]))
    assert torch.equal(block.edata(cites)['weight'], torch.tensor([5.0, 6.0, 7.0]))
    assert torch.equal(block.edge_index(writes), torch.tensor([[0, 0], [0, 1]]))
    assert torch.equal(block.edge_index(cites), torch.tensor([[0, 2, 1], [1, 1, 0]]))


def test_from_dgl_rejects_multi_relation_blocks(monkeypatch):
    dgl_module = _install_fake_dgl(monkeypatch)

    dgl_block = dgl_module.create_block(
        {
            ('author', 'writes', 'paper'): (torch.tensor([0]), torch.tensor([0])),
            ('paper', 'cites', 'paper'): (torch.tensor([0]), torch.tensor([0])),
        },
        num_src_nodes={'author': 1, 'paper': 1},
        num_dst_nodes={'paper': 1},
    )

    with pytest.raises(ValueError, match='single-relation'):
        Block.from_dgl(dgl_block)


def test_temporal_hetero_block_round_trips_time_attr_through_dgl_adapter(monkeypatch):
    _install_fake_dgl(monkeypatch)

    writes = ('author', 'writes', 'paper')
    cites = ('paper', 'cites', 'paper')
    graph = Graph.temporal(
        nodes={
            'author': {'x': torch.tensor([[10.0], [20.0]])},
            'paper': {'x': torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {
                'edge_index': torch.tensor([[0, 1, 1], [1, 0, 2]]),
                'timestamp': torch.tensor([1.0, 2.0, 3.0]),
            },
            cites: {
                'edge_index': torch.tensor([[0, 1, 2], [2, 2, 0]]),
                'timestamp': torch.tensor([4.0, 5.0, 6.0]),
            },
        },
        time_attr='timestamp',
    )

    block = graph.to_hetero_block({'paper': torch.tensor([0, 2])})

    dgl_block = block.to_dgl()
    restored = HeteroBlock.from_dgl(dgl_block)

    assert getattr(dgl_block, 'vgl_time_attr') == 'timestamp'
    assert restored.graph.schema.time_attr == 'timestamp'
    assert torch.equal(restored.edata(writes)['timestamp'], block.edata(writes)['timestamp'])
    assert torch.equal(restored.edata(cites)['timestamp'], block.edata(cites)['timestamp'])


def test_compat_exports_hetero_block_dgl_helpers(monkeypatch):
    _install_fake_dgl(monkeypatch)

    from vgl.compat import hetero_block_from_dgl, hetero_block_to_dgl

    writes = ('author', 'writes', 'paper')
    cites = ('paper', 'cites', 'paper')
    graph = Graph.hetero(
        nodes={
            'author': {'x': torch.tensor([[10.0], [20.0]])},
            'paper': {'x': torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {'edge_index': torch.tensor([[0, 1, 1], [1, 0, 2]])},
            cites: {'edge_index': torch.tensor([[0, 1, 2], [2, 2, 0]])},
        },
    )

    block = graph.to_hetero_block({'paper': torch.tensor([0, 2])})
    dgl_block = hetero_block_to_dgl(block)
    restored = hetero_block_from_dgl(dgl_block)

    assert isinstance(restored, HeteroBlock)
    assert restored.edge_types == block.edge_types
