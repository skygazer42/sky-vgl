import torch
import vgl.nn.encoders as encoder_impl

from vgl import Graph
from vgl.nn import GPSLayer
from vgl.nn import GraphConv
from vgl.nn import SGFormerEncoder
from vgl.nn import SGFormerEncoderLayer
from vgl.nn import GraphTransformerEncoder
from vgl.nn import GraphTransformerEncoderLayer
from vgl.nn import GraphormerEncoder
from vgl.nn import GraphormerEncoderLayer
from vgl.nn import NAGphormerEncoder


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]]),
        x=torch.randn(3, 4),
    )


def test_graph_transformer_encoder_layer_accepts_graph_input():
    encoder = GraphTransformerEncoderLayer(channels=4, heads=2, dropout=0.0)

    out = encoder(_graph())

    assert out.shape == (3, 4)


def test_graph_transformer_encoder_accepts_x_and_edge_index():
    graph = _graph()
    encoder = GraphTransformerEncoder(channels=4, num_layers=2, heads=2, dropout=0.0)

    out = encoder(graph.x, graph.edge_index)

    assert out.shape == (3, 4)


def test_graphormer_encoder_layer_accepts_graph_input():
    encoder = GraphormerEncoderLayer(channels=4, heads=2, max_distance=4, dropout=0.0)

    out = encoder(_graph())

    assert out.shape == (3, 4)


def test_shortest_path_buckets_match_expected_distances():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 2]])

    distances = encoder_impl._shortest_path_buckets(
        edge_index=edge_index,
        num_nodes=5,
        max_distance=2,
        device=edge_index.device,
    )

    assert torch.equal(
        distances,
        torch.tensor(
            [
                [0, 1, 2, 3, 3],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 3],
                [3, 2, 1, 0, 3],
                [3, 3, 3, 3, 0],
            ]
        ),
    )


def test_graphormer_encoder_layer_avoids_tensor_tolist(monkeypatch):
    encoder = GraphormerEncoderLayer(channels=4, heads=2, max_distance=4, dropout=0.0)

    def fail_tolist(self):
        raise AssertionError("Graphormer shortest-path buckets should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    out = encoder(_graph())

    assert out.shape == (3, 4)


def test_graphormer_encoder_layer_avoids_tensor_item(monkeypatch):
    encoder = GraphormerEncoderLayer(channels=4, heads=2, max_distance=4, dropout=0.0)

    def fail_item(self):
        raise AssertionError("Graphormer shortest-path buckets should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    out = encoder(_graph())

    assert out.shape == (3, 4)


def test_graphormer_encoder_accepts_x_and_edge_index():
    graph = _graph()
    encoder = GraphormerEncoder(channels=4, num_layers=2, heads=2, max_distance=4, dropout=0.0)

    out = encoder(graph.x, graph.edge_index)

    assert out.shape == (3, 4)


def test_gps_layer_accepts_graph_input():
    encoder = GPSLayer(
        channels=4,
        local_gnn=GraphConv(in_channels=4, out_channels=4),
        heads=2,
        dropout=0.0,
    )

    out = encoder(_graph())

    assert out.shape == (3, 4)


def test_nagphormer_encoder_accepts_graph_input():
    encoder = NAGphormerEncoder(channels=4, num_layers=2, num_hops=2, heads=2, dropout=0.0)

    out = encoder(_graph())

    assert out.shape == (3, 4)


def test_sgformer_encoder_layer_accepts_graph_input():
    encoder = SGFormerEncoderLayer(channels=4, heads=2, alpha=0.5, dropout=0.0)

    out = encoder(_graph())

    assert out.shape == (3, 4)


def test_sgformer_encoder_accepts_x_and_edge_index():
    graph = _graph()
    encoder = SGFormerEncoder(channels=4, num_layers=2, heads=2, alpha=0.5, dropout=0.0)

    out = encoder(graph.x, graph.edge_index)

    assert out.shape == (3, 4)
