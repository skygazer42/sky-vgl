import torch

from vgl import Graph
from vgl.graph import GraphSchema
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore


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
