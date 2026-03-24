import torch

from vgl import Graph
from vgl.graph import GraphSchema
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore


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


HOMO_EDGE = ("node", "to", "node")
WRITES = ("author", "writes", "paper")


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
