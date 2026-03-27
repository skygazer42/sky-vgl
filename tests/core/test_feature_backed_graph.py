import torch

from vgl import Graph
from vgl.graph import GraphSchema
from vgl.sparse import to_coo
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore
from vgl.ops import in_subgraph, out_subgraph


class RecordingTensorStore:
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self.fetch_calls = []

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def fetch(self, index: torch.Tensor):
        captured = torch.as_tensor(index, dtype=torch.long).clone()
        self.fetch_calls.append(captured)
        return type("TensorSliceProxy", (), {"index": captured, "values": self._tensor[captured]})()


class RecordingGraphStore:
    def __init__(self, edges, *, num_nodes):
        self._store = InMemoryGraphStore(edges, num_nodes=num_nodes)
        self.edge_index_calls = []
        self.edge_count_calls = []

    @property
    def edge_types(self):
        return self._store.edge_types

    def num_nodes(self, node_type: str = "node") -> int:
        return self._store.num_nodes(node_type)

    def _resolve_edge_type(self, edge_type):
        if edge_type is not None:
            return tuple(edge_type)
        if HOMO_EDGE in self._store.edge_types:
            return HOMO_EDGE
        if len(self._store.edge_types) == 1:
            return next(iter(self._store.edge_types))
        raise KeyError("edge_type is required when multiple edge types exist")

    def edge_index(self, edge_type=None):
        resolved = self._resolve_edge_type(edge_type)
        self.edge_index_calls.append(resolved)
        return self._store.edge_index(resolved)

    def edge_count(self, edge_type=None) -> int:
        resolved = self._resolve_edge_type(edge_type)
        self.edge_count_calls.append(resolved)
        return int(self._store.edge_index(resolved).size(1))

    def adjacency(self, *, edge_type=None, layout="coo"):
        return self._store.adjacency(edge_type=edge_type, layout=layout)


HOMO_EDGE = ("node", "to", "node")
WRITES = ("author", "writes", "paper")


def _sparse_to_dense(sparse) -> torch.Tensor:
    coo = to_coo(sparse)
    dtype = torch.float32 if coo.values is None else coo.values.dtype
    dense = torch.zeros(coo.shape, dtype=dtype)
    if coo.nnz == 0:
        return dense
    values = torch.ones(coo.nnz, dtype=dtype) if coo.values is None else coo.values.cpu()
    dense[coo.row.cpu(), coo.col.cpu()] = values
    return dense


def test_graph_from_storage_resolves_homo_features_and_edges():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    y = torch.tensor([0, 1, 0])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weight = torch.tensor([0.1, 0.2, 0.3])
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ("x", "y")},
        edge_features={HOMO_EDGE: ("edge_index", "edge_weight")},
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): InMemoryTensorStore(x),
            ("node", "node", "y"): InMemoryTensorStore(y),
            ("edge", HOMO_EDGE, "edge_weight"): InMemoryTensorStore(edge_weight),
        }
    )
    graph_store = InMemoryGraphStore({HOMO_EDGE: edge_index}, num_nodes={"node": 3})

    graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)

    assert torch.equal(graph.x, x)
    assert torch.equal(graph.y, y)
    assert torch.equal(graph.edge_index, edge_index)
    assert torch.equal(graph.edata["edge_weight"], edge_weight)
    assert graph.adjacency().shape == (3, 3)


def test_graph_from_storage_preserves_hetero_edge_shapes():
    author_x = torch.tensor([[1.0], [2.0]])
    paper_x = torch.tensor([[3.0], [4.0], [5.0], [6.0]])
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 3]])
    schema = GraphSchema(
        node_types=("author", "paper"),
        edge_types=(WRITES,),
        node_features={"author": ("x",), "paper": ("x",)},
        edge_features={WRITES: ("edge_index",)},
    )
    feature_store = FeatureStore(
        {
            ("node", "author", "x"): InMemoryTensorStore(author_x),
            ("node", "paper", "x"): InMemoryTensorStore(paper_x),
        }
    )
    graph_store = InMemoryGraphStore({WRITES: edge_index}, num_nodes={"author": 2, "paper": 4})

    graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)

    adjacency = graph.adjacency(edge_type=WRITES)

    assert torch.equal(graph.nodes["author"].x, author_x)
    assert torch.equal(graph.nodes["paper"].x, paper_x)
    assert torch.equal(graph.edges[WRITES].edge_index, edge_index)
    assert adjacency.shape == (2, 4)


def test_graph_from_storage_fetches_features_lazily_and_caches_them():
    x_store = RecordingTensorStore(torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]))
    y_store = RecordingTensorStore(torch.tensor([0, 1, 0]))
    edge_weight_store = RecordingTensorStore(torch.tensor([0.1, 0.2, 0.3]))
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ("x", "y")},
        edge_features={HOMO_EDGE: ("edge_index", "edge_weight")},
    )
    feature_store = FeatureStore(
        {
            ("node", "node", "x"): x_store,
            ("node", "node", "y"): y_store,
            ("edge", HOMO_EDGE, "edge_weight"): edge_weight_store,
        }
    )
    graph_store = InMemoryGraphStore({HOMO_EDGE: torch.tensor([[0, 1, 2], [1, 2, 0]])}, num_nodes={"node": 3})

    graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)

    assert x_store.fetch_calls == []
    assert y_store.fetch_calls == []
    assert edge_weight_store.fetch_calls == []

    assert torch.equal(graph.x, torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]))
    assert len(x_store.fetch_calls) == 1
    assert torch.equal(x_store.fetch_calls[0], torch.tensor([0, 1, 2]))
    assert y_store.fetch_calls == []
    assert edge_weight_store.fetch_calls == []

    assert torch.equal(graph.x, torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]))
    assert len(x_store.fetch_calls) == 1

    assert torch.equal(graph.edata["edge_weight"], torch.tensor([0.1, 0.2, 0.3]))
    assert len(edge_weight_store.fetch_calls) == 1
    assert torch.equal(edge_weight_store.fetch_calls[0], torch.tensor([0, 1, 2]))

    assert torch.equal(graph.y, torch.tensor([0, 1, 0]))
    assert len(y_store.fetch_calls) == 1
    assert torch.equal(y_store.fetch_calls[0], torch.tensor([0, 1, 2]))


def test_graph_from_storage_fetches_edge_structure_lazily_and_caches_it():
    edge_weight_store = RecordingTensorStore(torch.tensor([0.1, 0.2, 0.3]))
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index", "edge_weight")},
    )
    feature_store = FeatureStore(
        {
            ("edge", HOMO_EDGE, "edge_weight"): edge_weight_store,
        }
    )
    graph_store = RecordingGraphStore(
        {HOMO_EDGE: torch.tensor([[0, 1, 2], [1, 2, 0]])},
        num_nodes={"node": 3},
    )

    graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)

    assert graph_store.edge_index_calls == []
    assert graph_store.edge_count_calls == []
    assert edge_weight_store.fetch_calls == []

    assert torch.equal(graph.edata["edge_weight"], torch.tensor([0.1, 0.2, 0.3]))
    assert graph_store.edge_index_calls == []
    assert graph_store.edge_count_calls == [HOMO_EDGE]
    assert len(edge_weight_store.fetch_calls) == 1
    assert torch.equal(edge_weight_store.fetch_calls[0], torch.tensor([0, 1, 2]))

    assert torch.equal(graph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))
    assert graph_store.edge_index_calls == [HOMO_EDGE]

    assert torch.equal(graph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))
    adjacency = graph.adjacency()
    assert graph_store.edge_index_calls == [HOMO_EDGE]
    assert adjacency.shape == (3, 3)


def test_graph_from_storage_retains_feature_store_context():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ("x",)},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    feature_store = FeatureStore({("node", "node", "x"): InMemoryTensorStore(torch.tensor([[1.0], [2.0]]))})
    graph_store = InMemoryGraphStore({HOMO_EDGE: torch.tensor([[0], [1]])}, num_nodes={"node": 2})

    graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)

    assert graph.feature_store is feature_store


def test_graph_from_storage_retains_graph_store_context():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph_store = InMemoryGraphStore({HOMO_EDGE: torch.tensor([[0], [1]])}, num_nodes={"node": 2})

    graph = Graph.from_storage(schema=schema, feature_store=FeatureStore({}), graph_store=graph_store)

    assert graph.graph_store is graph_store


def test_storage_backed_graph_transfer_preserves_graph_store_context():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph_store = InMemoryGraphStore({HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])}, num_nodes={"node": 4})
    graph = Graph.from_storage(schema=schema, feature_store=FeatureStore({}), graph_store=graph_store)

    moved = graph.to(device="cpu")

    assert moved.graph_store is graph_store

    pinned = graph.pin_memory()

    assert pinned.graph_store is graph_store


def test_featureless_storage_backed_homo_graph_preserves_adjacency_shape():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    adjacency = graph.adjacency()

    assert adjacency.shape == (4, 4)


def test_featureless_storage_backed_hetero_graph_preserves_adjacency_shape():
    schema = GraphSchema(
        node_types=("author", "paper"),
        edge_types=(WRITES,),
        node_features={"author": (), "paper": ()},
        edge_features={WRITES: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {WRITES: torch.tensor([[0], [1]])},
            num_nodes={"author": 3, "paper": 4},
        ),
    )

    adjacency = graph.adjacency(edge_type=WRITES)

    assert adjacency.shape == (3, 4)


def test_featureless_storage_backed_frontier_subgraphs_preserve_node_count():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph_store = InMemoryGraphStore(
        {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
        num_nodes={"node": 4},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=graph_store,
    )

    inbound = in_subgraph(graph, torch.tensor([0]))
    outbound = out_subgraph(graph, torch.tensor([0]))

    assert inbound.graph_store is graph_store
    assert outbound.graph_store is graph_store
    assert inbound.adjacency().shape == (4, 4)
    assert outbound.adjacency().shape == (4, 4)
    assert torch.equal(inbound.edge_index, torch.tensor([[1], [0]]))
    assert torch.equal(outbound.edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(inbound.edata["e_id"], torch.tensor([1]))
    assert torch.equal(outbound.edata["e_id"], torch.tensor([0]))


def test_featureless_storage_backed_reverse_preserves_node_count():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph_store = InMemoryGraphStore(
        {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
        num_nodes={"node": 4},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=graph_store,
    )

    reversed_graph = graph.reverse()

    assert reversed_graph.graph_store is graph_store
    assert reversed_graph.adjacency().shape == (4, 4)
    assert torch.equal(reversed_graph.edge_index, torch.tensor([[1, 0], [0, 1]]))
    assert torch.equal(reversed_graph.edata["e_id"], torch.tensor([0, 1]))


def test_featureless_storage_backed_adjacency_queries_preserve_declared_node_space():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    assert torch.equal(graph.successors(3), torch.empty(0, dtype=torch.long))
    assert torch.equal(graph.predecessors(3), torch.empty(0, dtype=torch.long))
    assert torch.equal(graph.in_edges(torch.tensor([3]), form="eid"), torch.empty(0, dtype=torch.long))
    assert torch.equal(graph.out_edges(torch.tensor([3]), form="eid"), torch.empty(0, dtype=torch.long))


def test_featureless_storage_backed_in_degrees_and_out_degrees_preserve_declared_node_space():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    assert graph.in_degrees(3) == 0
    assert graph.out_degrees(3) == 0
    assert torch.equal(graph.in_degrees(), torch.tensor([1, 1, 0, 0]))
    assert torch.equal(graph.out_degrees(torch.tensor([0, 3])), torch.tensor([1, 0]))


def test_featureless_storage_backed_cardinality_and_all_edges_preserve_declared_node_space():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    assert graph.num_nodes() == 4
    assert graph.number_of_nodes() == 4
    assert graph.num_edges() == 2
    assert graph.number_of_edges() == 2
    src, dst, eids = graph.all_edges(form="all")

    assert torch.equal(src, torch.tensor([0, 1]))
    assert torch.equal(dst, torch.tensor([1, 0]))
    assert torch.equal(eids, torch.tensor([0, 1]))


def test_featureless_storage_backed_incidence_preserves_declared_node_space():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    incidence = graph.inc("in")

    assert incidence.shape == (4, 2)
    assert torch.equal(
        _sparse_to_dense(incidence),
        torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_featureless_storage_backed_adj_tensors_preserve_declared_node_space():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    crow_indices, col_indices, csr_eids = graph.adj_tensors("csr")
    ccol_indices, row_indices, csc_eids = graph.adj_tensors("csc")

    assert torch.equal(crow_indices, torch.tensor([0, 1, 2, 2, 2]))
    assert torch.equal(col_indices, torch.tensor([1, 0]))
    assert torch.equal(csr_eids, torch.tensor([0, 1]))
    assert torch.equal(ccol_indices, torch.tensor([0, 1, 2, 2, 2]))
    assert torch.equal(row_indices, torch.tensor([1, 0]))
    assert torch.equal(csc_eids, torch.tensor([1, 0]))


def test_featureless_storage_backed_adj_preserves_declared_node_space():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    adjacency = graph.adj(layout="csr")

    assert adjacency.shape == (4, 4)
    assert torch.equal(adjacency.crow_indices, torch.tensor([0, 1, 2, 2, 2]))
    assert torch.equal(
        _sparse_to_dense(adjacency),
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_featureless_storage_backed_adj_external_preserves_declared_node_space():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    adjacency = graph.adj_external()

    assert adjacency.layout is torch.sparse_coo
    assert tuple(adjacency.size()) == (4, 4)
    assert torch.equal(adjacency._indices(), torch.tensor([[0, 1], [1, 0]]))


def test_featureless_storage_backed_graph_formats_preserve_status_model():
    schema = GraphSchema(
        node_types=("node",),
        edge_types=(HOMO_EDGE,),
        node_features={"node": ()},
        edge_features={HOMO_EDGE: ("edge_index",)},
    )
    graph = Graph.from_storage(
        schema=schema,
        feature_store=FeatureStore({}),
        graph_store=InMemoryGraphStore(
            {HOMO_EDGE: torch.tensor([[0, 1], [1, 0]])},
            num_nodes={"node": 4},
        ),
    )

    assert graph.formats() == {"created": ["coo"], "not created": ["csr", "csc"]}

    graph.create_formats_()

    assert graph.formats() == {"created": ["coo", "csr", "csc"], "not created": []}
